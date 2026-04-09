# API Gateway Implementation Plan (Phase 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `claw-gateway` binary that exposes the local Ollama server to external clients over HTTPS with API key authentication, per-key rate limiting, and endpoint filtering.

**Architecture:** The gateway is a new `gateway` crate in the workspace producing a separate binary (`claw-gateway`). It uses axum as the HTTP framework with tower middleware layers stacked in order: TLS termination (rustls) -> API key auth -> per-key rate limiting -> endpoint filtering -> reverse proxy to Ollama at `127.0.0.1:11434`. API keys are stored in `~/.claw/gateway/keys.json` and managed via CLI subcommands.

**Tech Stack:** axum 0.8, tower 0.5, tower-http 0.6, rustls 0.23, rcgen (self-signed certs), hyper 1, hyper-util, http-body-util, clap 4, serde/serde_json, tokio, sha2, rand, chrono

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `crates/gateway/Cargo.toml` | Crate manifest with all dependencies |
| Create | `crates/gateway/src/main.rs` | Binary entry point, clap CLI (serve, key create/list/revoke) |
| Create | `crates/gateway/src/auth.rs` | Bearer token extraction + validation middleware |
| Create | `crates/gateway/src/proxy.rs` | Reverse proxy forwarding to Ollama, OpenAI-compat translation |
| Create | `crates/gateway/src/ratelimit.rs` | Per-key token-bucket rate limiter |
| Create | `crates/gateway/src/tls.rs` | TLS config: self-signed cert generation or file-based certs |
| Create | `crates/gateway/src/keys.rs` | Key CRUD: generate, list, revoke, load/save keys.json |
| Create | `crates/gateway/src/filter.rs` | Endpoint allowlist/blocklist enforcement |
| Create | `crates/gateway/tests/auth_test.rs` | Unit tests for auth middleware |
| Create | `crates/gateway/tests/filter_test.rs` | Unit tests for endpoint filtering |
| Create | `crates/gateway/tests/keys_test.rs` | Unit tests for key management CRUD |
| Create | `crates/gateway/tests/ratelimit_test.rs` | Unit tests for rate limiting |
| Create | `crates/gateway/tests/integration.rs` | Integration test: full request flow (requires Ollama) |
| Modify | `rust/Cargo.toml` | Already auto-included via `members = ["crates/*"]` -- no change needed |

---

### Task 1: Create gateway crate scaffold and CLI

**Files:** `crates/gateway/Cargo.toml`, `crates/gateway/src/main.rs`

- [ ] **Step 1: Create `Cargo.toml`**

Create `rust/crates/gateway/Cargo.toml`:

```toml
[package]
name = "gateway"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[[bin]]
name = "claw-gateway"
path = "src/main.rs"

[dependencies]
axum = "0.8"
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.6", features = ["cors", "trace"] }
rustls = "0.23"
rustls-pemfile = "2"
rcgen = "0.13"
tokio-rustls = "0.26"
hyper = { version = "1", features = ["full"] }
hyper-util = { version = "0.1", features = ["client-legacy", "http1", "http2", "tokio"] }
http-body-util = "0.1"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json.workspace = true
tokio = { version = "1", features = ["rt-multi-thread", "signal", "macros", "net", "time", "sync"] }
sha2 = "0.10"
rand = "0.8"
chrono = { version = "0.4", features = ["serde"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
base64 = "0.22"

[lints]
workspace = true
```

- [ ] **Step 2: Create `main.rs` with clap CLI skeleton**

Create `rust/crates/gateway/src/main.rs`:

```rust
mod auth;
mod filter;
mod keys;
mod proxy;
mod ratelimit;
mod tls;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use clap::{Parser, Subcommand};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use crate::keys::KeyStore;
use crate::ratelimit::RateLimiterMap;

#[derive(Parser)]
#[command(name = "claw-gateway", about = "Claw Code API Gateway for Ollama")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the gateway server
    Serve {
        /// Port to listen on
        #[arg(long, default_value = "8443")]
        port: u16,

        /// Ollama backend URL
        #[arg(long, default_value = "http://127.0.0.1:11434")]
        ollama: String,

        /// TLS mode: "self-signed", "files", or "none"
        #[arg(long, default_value = "self-signed")]
        tls: String,

        /// Path to TLS certificate (when --tls=files)
        #[arg(long)]
        cert: Option<String>,

        /// Path to TLS private key (when --tls=files)
        #[arg(long)]
        key_file: Option<String>,
    },
    /// Manage API keys
    Key {
        #[command(subcommand)]
        action: KeyAction,
    },
}

#[derive(Subcommand)]
enum KeyAction {
    /// Create a new API key
    Create {
        /// Human-readable name for this key
        #[arg(long)]
        name: String,

        /// Rate limit (requests per minute)
        #[arg(long, default_value = "30")]
        rate_limit: u32,
    },
    /// List all API keys
    List,
    /// Revoke an API key by ID
    Revoke {
        /// Key ID to revoke (e.g. "key_01")
        id: String,
    },
}

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub key_store: Arc<RwLock<KeyStore>>,
    pub rate_limiters: Arc<RwLock<RateLimiterMap>>,
    pub ollama_url: String,
    pub http_client: hyper_util::client::legacy::Client<
        hyper_util::client::legacy::connect::HttpConnector,
        axum::body::Body,
    >,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("gateway=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            port,
            ollama,
            tls: tls_mode,
            cert,
            key_file,
        } => {
            let key_store = Arc::new(RwLock::new(KeyStore::load_or_create()?));
            let rate_limiters = Arc::new(RwLock::new(RateLimiterMap::new()));

            let http_client = hyper_util::client::legacy::Client::builder(
                hyper_util::rt::TokioExecutor::new(),
            )
            .build_http();

            let state = AppState {
                key_store,
                rate_limiters,
                ollama_url: ollama.clone(),
                http_client,
            };

            let app = build_router(state);
            let addr = SocketAddr::from(([0, 0, 0, 0], port));

            tracing::info!("Gateway listening on {addr}, proxying to {ollama}");

            match tls_mode.as_str() {
                "none" => {
                    let listener = tokio::net::TcpListener::bind(addr).await?;
                    axum::serve(listener, app)
                        .with_graceful_shutdown(shutdown_signal())
                        .await?;
                }
                "self-signed" => {
                    tls::serve_with_self_signed_tls(addr, app).await?;
                }
                "files" => {
                    let cert_path =
                        cert.expect("--cert is required when --tls=files");
                    let key_path =
                        key_file.expect("--key-file is required when --tls=files");
                    tls::serve_with_tls_files(addr, app, &cert_path, &key_path).await?;
                }
                other => {
                    anyhow::bail!("Unknown TLS mode: {other}. Use self-signed, files, or none.");
                }
            }
        }
        Commands::Key { action } => match action {
            KeyAction::Create { name, rate_limit } => {
                let mut store = KeyStore::load_or_create()?;
                let key = store.create_key(&name, rate_limit)?;
                println!("Created API key:");
                println!("  ID:     {}", key.id);
                println!("  Name:   {}", key.name);
                println!("  Secret: {}", key.secret);
                println!("  Limit:  {} req/min", key.rate_limit);
                println!("\nStore this secret -- it cannot be retrieved later.");
            }
            KeyAction::List => {
                let store = KeyStore::load_or_create()?;
                store.print_table();
            }
            KeyAction::Revoke { id } => {
                let mut store = KeyStore::load_or_create()?;
                store.revoke(&id)?;
                println!("Key {id} revoked.");
            }
        },
    }

    Ok(())
}

fn build_router(state: AppState) -> Router {
    use axum::routing::{any, get};

    Router::new()
        // Health endpoint (no auth required)
        .route("/health", get(|| async { "OK" }))
        // All Ollama proxy routes go through the middleware stack
        .route("/api/{*path}", any(proxy::proxy_handler))
        .route("/v1/chat/completions", axum::routing::post(proxy::openai_compat_handler))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            ratelimit::rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .with_state(state)
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!("Shutdown signal received");
}
```

- [ ] **Step 3: Verify scaffold compiles**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p gateway`

Create stub files first (empty module bodies) so compilation succeeds -- the real implementations come in subsequent tasks. Create stub files for `auth.rs`, `proxy.rs`, `ratelimit.rs`, `tls.rs`, `keys.rs`, `filter.rs` with minimal contents to satisfy `mod` declarations.

---

### Task 2: Key management (keys.rs)

**Files:** `crates/gateway/src/keys.rs`, `crates/gateway/tests/keys_test.rs`

- [ ] **Step 1: Write failing tests for key CRUD**

Create `rust/crates/gateway/tests/keys_test.rs`:

```rust
use gateway::keys::KeyStore;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn create_key_generates_prefixed_secret() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("test-laptop", 30).unwrap();

    assert!(key.secret.starts_with("claw-sk-"));
    assert_eq!(key.name, "test-laptop");
    assert_eq!(key.rate_limit, 30);
    assert!(key.enabled);
}

#[test]
fn revoke_key_disables_it() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("ephemeral", 10).unwrap();
    let id = key.id.clone();

    store.revoke(&id).unwrap();
    assert!(!store.find_by_id(&id).unwrap().enabled);
}

#[test]
fn validate_secret_returns_matching_key() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("my-key", 30).unwrap();
    let secret = key.secret.clone();

    let found = store.validate_secret(&secret);
    assert!(found.is_some());
    assert_eq!(found.unwrap().name, "my-key");
}

#[test]
fn validate_revoked_key_returns_none() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("revoked", 30).unwrap();
    let secret = key.secret.clone();
    store.revoke(&key.id).unwrap();

    let found = store.validate_secret(&secret);
    assert!(found.is_none());
}
```

- [ ] **Step 2: Implement `keys.rs`**

Create `rust/crates/gateway/src/keys.rs`:

```rust
use std::fs;
use std::path::{Path, PathBuf};

use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A single API key entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    /// The raw secret token (`claw-sk-...`). Stored in plaintext in keys.json
    /// (local file, single-user machine). For production use, store a hash instead.
    pub secret: String,
    /// SHA-256 hash of the secret for O(1) lookup.
    #[serde(default)]
    pub secret_hash: String,
    pub rate_limit: u32,
    pub enabled: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeysFile {
    pub keys: Vec<ApiKey>,
}

/// In-memory key store backed by a JSON file.
#[derive(Debug, Clone)]
pub struct KeyStore {
    path: PathBuf,
    pub keys: Vec<ApiKey>,
    next_id: u32,
}

impl KeyStore {
    /// Load from the default location: `~/.claw/gateway/keys.json`.
    pub fn load_or_create() -> anyhow::Result<Self> {
        let home = dirs_or_home()?;
        let path = home.join(".claw").join("gateway").join("keys.json");
        Self::load_from_path(&path)
    }

    /// Load from an explicit path (useful for tests).
    pub fn load_from_path(path: &Path) -> anyhow::Result<Self> {
        if path.exists() {
            let data = fs::read_to_string(path)?;
            let file: KeysFile = serde_json::from_str(&data)?;
            let next_id = file
                .keys
                .iter()
                .filter_map(|k| k.id.strip_prefix("key_").and_then(|n| n.parse::<u32>().ok()))
                .max()
                .unwrap_or(0)
                + 1;
            Ok(Self {
                path: path.to_path_buf(),
                keys: file.keys,
                next_id,
            })
        } else {
            Ok(Self {
                path: path.to_path_buf(),
                keys: Vec::new(),
                next_id: 1,
            })
        }
    }

    /// Create a new API key, persist to disk, return the key.
    pub fn create_key(&mut self, name: &str, rate_limit: u32) -> anyhow::Result<ApiKey> {
        let id = format!("key_{:02}", self.next_id);
        self.next_id += 1;

        let secret = generate_secret();
        let secret_hash = hash_secret(&secret);
        let created_at = chrono::Utc::now().to_rfc3339();

        let key = ApiKey {
            id,
            name: name.to_string(),
            secret,
            secret_hash,
            rate_limit,
            enabled: true,
            created_at,
        };

        self.keys.push(key.clone());
        self.save()?;
        Ok(key)
    }

    /// Revoke a key by ID.
    pub fn revoke(&mut self, id: &str) -> anyhow::Result<()> {
        let key = self
            .keys
            .iter_mut()
            .find(|k| k.id == id)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {id}"))?;
        key.enabled = false;
        self.save()
    }

    /// Find a key by ID.
    pub fn find_by_id(&self, id: &str) -> Option<&ApiKey> {
        self.keys.iter().find(|k| k.id == id)
    }

    /// Validate a Bearer secret. Returns the key if found AND enabled.
    pub fn validate_secret(&self, secret: &str) -> Option<&ApiKey> {
        let hash = hash_secret(secret);
        self.keys
            .iter()
            .find(|k| k.secret_hash == hash && k.enabled)
    }

    /// Print a human-readable table of all keys.
    pub fn print_table(&self) {
        println!("{:<10} {:<20} {:<8} {:<10}", "ID", "Name", "Limit", "Status");
        println!("{}", "-".repeat(52));
        for key in &self.keys {
            let status = if key.enabled { "active" } else { "revoked" };
            println!(
                "{:<10} {:<20} {:<8} {:<10}",
                key.id, key.name, key.rate_limit, status
            );
        }
    }

    fn save(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = KeysFile {
            keys: self.keys.clone(),
        };
        let json = serde_json::to_string_pretty(&file)?;
        fs::write(&self.path, json)?;
        Ok(())
    }
}

fn generate_secret() -> String {
    let mut rng = rand::thread_rng();
    let bytes: [u8; 32] = rng.gen();
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    // Use base64 crate -- imported via `use base64::Engine` in the actual code.
    format!("claw-sk-{encoded}")
}

fn hash_secret(secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn dirs_or_home() -> anyhow::Result<PathBuf> {
    dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))
}
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p gateway --test keys_test -- --nocapture`

---

### Task 3: Endpoint filtering (filter.rs)

**Files:** `crates/gateway/src/filter.rs`, `crates/gateway/tests/filter_test.rs`

- [ ] **Step 1: Write failing tests**

Create `rust/crates/gateway/tests/filter_test.rs`:

```rust
use gateway::filter::EndpointFilter;

#[test]
fn allows_chat_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/api/chat"));
}

#[test]
fn allows_generate_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/api/generate"));
}

#[test]
fn allows_show_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/api/show"));
}

#[test]
fn allows_ps_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("GET", "/api/ps"));
}

#[test]
fn allows_tags_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("GET", "/api/tags"));
}

#[test]
fn allows_openai_compat_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/v1/chat/completions"));
}

#[test]
fn blocks_delete_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("DELETE", "/api/delete"));
}

#[test]
fn blocks_create_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/create"));
}

#[test]
fn blocks_pull_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/pull"));
}

#[test]
fn blocks_push_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/push"));
}

#[test]
fn blocks_unknown_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/something-new"));
}
```

- [ ] **Step 2: Implement `filter.rs`**

Create `rust/crates/gateway/src/filter.rs`:

```rust
/// Allowlist-based endpoint filter. Only explicitly listed routes are permitted.
/// This is a security-critical component -- default-deny.
#[derive(Debug, Clone)]
pub struct EndpointFilter {
    allowed: Vec<AllowedRoute>,
}

#[derive(Debug, Clone)]
struct AllowedRoute {
    method: &'static str,
    path: &'static str,
}

impl Default for EndpointFilter {
    fn default() -> Self {
        Self {
            allowed: vec![
                AllowedRoute { method: "POST", path: "/api/chat" },
                AllowedRoute { method: "POST", path: "/api/generate" },
                AllowedRoute { method: "POST", path: "/api/show" },
                AllowedRoute { method: "GET", path: "/api/ps" },
                AllowedRoute { method: "GET", path: "/api/tags" },
                AllowedRoute { method: "POST", path: "/v1/chat/completions" },
            ],
        }
    }
}

impl EndpointFilter {
    /// Returns `true` if the given method + path combination is on the allowlist.
    pub fn is_allowed(&self, method: &str, path: &str) -> bool {
        // Normalize: strip trailing slash
        let path = path.trim_end_matches('/');
        self.allowed
            .iter()
            .any(|r| r.method.eq_ignore_ascii_case(method) && r.path == path)
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p gateway --test filter_test -- --nocapture`

---

### Task 4: Authentication middleware (auth.rs)

**Files:** `crates/gateway/src/auth.rs`, `crates/gateway/tests/auth_test.rs`

- [ ] **Step 1: Write failing tests**

Create `rust/crates/gateway/tests/auth_test.rs`:

```rust
use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
    routing::get,
    middleware,
};
use tower::ServiceExt;
use std::sync::Arc;
use tokio::sync::RwLock;

// These tests need the full AppState, so they are integration-level.
// We test the auth extraction logic in isolation here.

#[tokio::test]
async fn missing_auth_header_returns_401() {
    let app = test_app().await;
    let req = Request::builder()
        .uri("/api/chat")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn invalid_bearer_token_returns_401() {
    let app = test_app().await;
    let req = Request::builder()
        .uri("/api/chat")
        .header("Authorization", "Bearer invalid-token")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn valid_bearer_token_passes_through() {
    let (app, secret) = test_app_with_key().await;
    let req = Request::builder()
        .uri("/test")
        .header("Authorization", format!("Bearer {secret}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// Helper: builds a minimal app with auth middleware and no real keys
async fn test_app() -> Router {
    use gateway::keys::KeyStore;
    use gateway::auth::auth_middleware;
    use gateway::AppState;
    use gateway::ratelimit::RateLimiterMap;

    let store = KeyStore::load_from_path(
        &std::env::temp_dir().join("claw-test-no-keys.json"),
    ).unwrap();

    let state = AppState {
        key_store: Arc::new(RwLock::new(store)),
        rate_limiters: Arc::new(RwLock::new(RateLimiterMap::new())),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        http_client: hyper_util::client::legacy::Client::builder(
            hyper_util::rt::TokioExecutor::new(),
        ).build_http(),
    };

    Router::new()
        .route("/api/chat", get(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state)
}

// Helper: builds an app with one valid key
async fn test_app_with_key() -> (Router, String) {
    use gateway::keys::KeyStore;
    use gateway::auth::auth_middleware;
    use gateway::AppState;
    use gateway::ratelimit::RateLimiterMap;

    let tmp = std::env::temp_dir().join("claw-test-with-key.json");
    let mut store = KeyStore::load_from_path(&tmp).unwrap();
    let key = store.create_key("test", 100).unwrap();
    let secret = key.secret.clone();

    let state = AppState {
        key_store: Arc::new(RwLock::new(store)),
        rate_limiters: Arc::new(RwLock::new(RateLimiterMap::new())),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        http_client: hyper_util::client::legacy::Client::builder(
            hyper_util::rt::TokioExecutor::new(),
        ).build_http(),
    };

    let app = Router::new()
        .route("/test", get(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state);

    (app, secret)
}
```

- [ ] **Step 2: Implement `auth.rs`**

Create `rust/crates/gateway/src/auth.rs`:

```rust
use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::AppState;

/// Extension type injected into the request after successful auth.
/// Downstream handlers can extract this to know which key made the request.
#[derive(Debug, Clone)]
pub struct AuthenticatedKey {
    pub key_id: String,
    pub key_name: String,
    pub rate_limit: u32,
}

/// Axum middleware: extracts `Authorization: Bearer <token>`, validates against
/// the key store, and injects `AuthenticatedKey` into request extensions.
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Skip auth for health endpoint
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => {
            tracing::warn!("Missing or malformed Authorization header from {:?}", req.uri());
            return (
                StatusCode::UNAUTHORIZED,
                "Missing or invalid Authorization header. Use: Bearer <api-key>",
            )
                .into_response();
        }
    };

    let store = state.key_store.read().await;
    match store.validate_secret(token) {
        Some(key) => {
            let auth_key = AuthenticatedKey {
                key_id: key.id.clone(),
                key_name: key.name.clone(),
                rate_limit: key.rate_limit,
            };
            tracing::info!(key_id = %auth_key.key_id, "Authenticated request");
            req.extensions_mut().insert(auth_key);
            drop(store); // release read lock before proceeding
            next.run(req).await
        }
        None => {
            tracing::warn!("Invalid API key attempt");
            (StatusCode::UNAUTHORIZED, "Invalid API key").into_response()
        }
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p gateway --test auth_test -- --nocapture`

---

### Task 5: Per-key rate limiting (ratelimit.rs)

**Files:** `crates/gateway/src/ratelimit.rs`, `crates/gateway/tests/ratelimit_test.rs`

- [ ] **Step 1: Write failing tests**

Create `rust/crates/gateway/tests/ratelimit_test.rs`:

```rust
use gateway::ratelimit::{RateLimiterMap, RateLimitResult};
use std::time::Duration;

#[test]
fn allows_requests_within_limit() {
    let mut map = RateLimiterMap::new();
    for _ in 0..5 {
        let result = map.check_and_consume("key_01", 10); // 10 req/min
        assert!(matches!(result, RateLimitResult::Allowed));
    }
}

#[test]
fn rejects_requests_over_limit() {
    let mut map = RateLimiterMap::new();
    // Consume all 3 tokens
    for _ in 0..3 {
        map.check_and_consume("key_01", 3);
    }
    let result = map.check_and_consume("key_01", 3);
    assert!(matches!(result, RateLimitResult::Limited { .. }));
}

#[test]
fn different_keys_have_independent_limits() {
    let mut map = RateLimiterMap::new();
    // Exhaust key_01
    for _ in 0..2 {
        map.check_and_consume("key_01", 2);
    }
    let r1 = map.check_and_consume("key_01", 2);
    assert!(matches!(r1, RateLimitResult::Limited { .. }));

    // key_02 should still work
    let r2 = map.check_and_consume("key_02", 2);
    assert!(matches!(r2, RateLimitResult::Allowed));
}
```

- [ ] **Step 2: Implement `ratelimit.rs`**

Create `rust/crates/gateway/src/ratelimit.rs`:

```rust
use std::collections::HashMap;
use std::time::Instant;

use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::auth::AuthenticatedKey;
use crate::AppState;

/// Result of a rate limit check.
#[derive(Debug)]
pub enum RateLimitResult {
    Allowed,
    Limited { retry_after_secs: u64 },
}

/// Token bucket for a single key.
#[derive(Debug, Clone)]
struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(requests_per_minute: u32) -> Self {
        let max = requests_per_minute as f64;
        Self {
            tokens: max,
            max_tokens: max,
            refill_rate: max / 60.0,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self) -> RateLimitResult {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            RateLimitResult::Allowed
        } else {
            let wait = (1.0 - self.tokens) / self.refill_rate;
            RateLimitResult::Limited {
                retry_after_secs: wait.ceil() as u64,
            }
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }
}

/// Map of key_id -> token bucket.
#[derive(Debug)]
pub struct RateLimiterMap {
    buckets: HashMap<String, TokenBucket>,
}

impl RateLimiterMap {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Check rate limit for a key. Creates a bucket on first access.
    pub fn check_and_consume(&mut self, key_id: &str, requests_per_minute: u32) -> RateLimitResult {
        let bucket = self
            .buckets
            .entry(key_id.to_string())
            .or_insert_with(|| TokenBucket::new(requests_per_minute));
        bucket.try_consume()
    }
}

/// Axum middleware: enforces per-key rate limits.
/// Must run AFTER auth middleware (needs `AuthenticatedKey` in extensions).
pub async fn rate_limit_middleware(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Skip rate limiting for health endpoint
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let auth_key = match req.extensions().get::<AuthenticatedKey>() {
        Some(k) => k.clone(),
        None => {
            // Auth middleware should have rejected this already, but be safe.
            return (StatusCode::UNAUTHORIZED, "Not authenticated").into_response();
        }
    };

    let result = {
        let mut limiters = state.rate_limiters.write().await;
        limiters.check_and_consume(&auth_key.key_id, auth_key.rate_limit)
    };

    match result {
        RateLimitResult::Allowed => next.run(req).await,
        RateLimitResult::Limited { retry_after_secs } => {
            tracing::warn!(key_id = %auth_key.key_id, "Rate limited");
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("Retry-After", retry_after_secs.to_string())],
                format!("Rate limit exceeded. Retry after {retry_after_secs}s."),
            )
                .into_response()
        }
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p gateway --test ratelimit_test -- --nocapture`

---

### Task 6: Reverse proxy + OpenAI-compat endpoint (proxy.rs)

**Files:** `crates/gateway/src/proxy.rs`

- [ ] **Step 1: Implement endpoint filter check in proxy**

Create `rust/crates/gateway/src/proxy.rs`:

```rust
use axum::{
    body::Body,
    extract::{Request, State},
    http::{uri::Uri, StatusCode},
    response::{IntoResponse, Response},
};
use http_body_util::BodyExt;

use crate::filter::EndpointFilter;
use crate::AppState;

/// Main proxy handler for `/api/*path`.
/// Checks the endpoint filter, then forwards to Ollama.
pub async fn proxy_handler(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Response {
    let method = req.method().as_str().to_uppercase();
    let path = req.uri().path().to_string();

    // Endpoint filtering (default-deny)
    let filter = EndpointFilter::default();
    if !filter.is_allowed(&method, &path) {
        tracing::warn!(method = %method, path = %path, "Blocked endpoint");
        return (
            StatusCode::FORBIDDEN,
            format!("Endpoint not allowed: {method} {path}"),
        )
            .into_response();
    }

    forward_to_ollama(state, req).await
}

/// OpenAI-compatible `/v1/chat/completions` handler.
/// Ollama natively supports this endpoint, so we just proxy it through.
pub async fn openai_compat_handler(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Response {
    // The filter already allows POST /v1/chat/completions (checked by middleware
    // ordering if filter is a layer, or we check here explicitly).
    forward_to_ollama(state, req).await
}

/// Forward a request to the Ollama backend.
async fn forward_to_ollama(state: AppState, req: Request<Body>) -> Response {
    let ollama_url = &state.ollama_url;
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");

    let target_uri = format!("{ollama_url}{path_and_query}");
    let uri: Uri = match target_uri.parse() {
        Ok(u) => u,
        Err(e) => {
            tracing::error!("Failed to parse target URI: {e}");
            return (StatusCode::BAD_GATEWAY, "Invalid upstream URI").into_response();
        }
    };

    // Build the proxied request
    let (mut parts, body) = req.into_parts();
    parts.uri = uri;

    // Remove hop-by-hop headers
    parts.headers.remove("host");
    parts.headers.remove("authorization"); // Don't forward our auth to Ollama

    let proxied_req = Request::from_parts(parts, body);

    // Forward to Ollama
    match state.http_client.request(proxied_req).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            let body = Body::new(body);
            Response::from_parts(parts, body)
        }
        Err(e) => {
            tracing::error!("Proxy error: {e}");
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to connect to Ollama: {e}"),
            )
                .into_response()
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p gateway`

---

### Task 7: TLS setup (tls.rs)

**Files:** `crates/gateway/src/tls.rs`

- [ ] **Step 1: Implement self-signed cert generation and TLS server bootstrap**

Create `rust/crates/gateway/src/tls.rs`:

```rust
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use rustls::ServerConfig;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

/// Serve with an auto-generated self-signed certificate.
/// Certificate is cached in `~/.claw/gateway/tls/` for reuse.
pub async fn serve_with_self_signed_tls(addr: SocketAddr, app: Router) -> anyhow::Result<()> {
    let tls_dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot find home dir"))?
        .join(".claw")
        .join("gateway")
        .join("tls");
    fs::create_dir_all(&tls_dir)?;

    let cert_path = tls_dir.join("self-signed.crt");
    let key_path = tls_dir.join("self-signed.key");

    if !cert_path.exists() || !key_path.exists() {
        tracing::info!("Generating self-signed TLS certificate...");
        generate_self_signed(&cert_path, &key_path)?;
    }

    serve_with_tls_files(addr, app, cert_path.to_str().unwrap(), key_path.to_str().unwrap())
        .await
}

/// Serve with TLS using provided certificate and key files.
pub async fn serve_with_tls_files(
    addr: SocketAddr,
    app: Router,
    cert_path: &str,
    key_path: &str,
) -> anyhow::Result<()> {
    let cert_pem = fs::read(cert_path)?;
    let key_pem = fs::read(key_path)?;

    let certs = rustls_pemfile::certs(&mut &cert_pem[..])
        .collect::<Result<Vec<_>, _>>()?;
    let key = rustls_pemfile::private_key(&mut &key_pem[..])?
        .ok_or_else(|| anyhow::anyhow!("No private key found in {key_path}"))?;

    let tls_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;

    let acceptor = TlsAcceptor::from(Arc::new(tls_config));
    let listener = TcpListener::bind(addr).await?;

    tracing::info!("TLS server listening on {addr}");

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        let acceptor = acceptor.clone();
        let app = app.clone();

        tokio::spawn(async move {
            match acceptor.accept(stream).await {
                Ok(tls_stream) => {
                    let io = hyper_util::rt::TokioIo::new(tls_stream);
                    let service = hyper::service::service_fn(move |req| {
                        let app = app.clone();
                        async move {
                            let resp = tower::ServiceExt::oneshot(app, req).await;
                            resp.map_err(|e| match e {})
                        }
                    });
                    if let Err(e) =
                        hyper_util::server::conn::auto::Builder::new(
                            hyper_util::rt::TokioExecutor::new(),
                        )
                        .serve_connection(io, service)
                        .await
                    {
                        tracing::error!("Connection error from {peer_addr}: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("TLS handshake failed from {peer_addr}: {e}");
                }
            }
        });
    }
}

/// Generate a self-signed certificate using rcgen.
fn generate_self_signed(cert_path: &PathBuf, key_path: &PathBuf) -> anyhow::Result<()> {
    use rcgen::{CertificateParams, KeyPair};

    let mut params = CertificateParams::new(vec![
        "localhost".to_string(),
        "0.0.0.0".to_string(),
    ])?;
    params.distinguished_name.push(
        rcgen::DnType::CommonName,
        rcgen::DnValue::Utf8String("claw-gateway".to_string()),
    );

    let key_pair = KeyPair::generate()?;
    let cert = params.self_signed(&key_pair)?;

    fs::write(cert_path, cert.pem())?;
    fs::write(key_path, key_pair.serialize_pem())?;

    tracing::info!("Self-signed cert written to {}", cert_path.display());
    Ok(())
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p gateway`

---

### Task 8: Integration test (end-to-end)

**Files:** `crates/gateway/tests/integration.rs`

- [ ] **Step 1: Write integration test**

Create `rust/crates/gateway/tests/integration.rs`:

```rust
//! Integration tests for the full gateway stack.
//! These tests start the gateway on a random port with TLS disabled
//! and verify the full request flow.
//!
//! NOTE: Tests that proxy to Ollama require a running Ollama instance.
//! Those are gated behind `#[ignore]` and run with `cargo test -- --ignored`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use std::sync::Arc;
use tokio::sync::RwLock;

use gateway::keys::KeyStore;
use gateway::ratelimit::RateLimiterMap;
use gateway::AppState;

async fn make_state_with_key() -> (AppState, String) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("integration-test", 60).unwrap();
    let secret = key.secret.clone();

    let state = AppState {
        key_store: Arc::new(RwLock::new(store)),
        rate_limiters: Arc::new(RwLock::new(RateLimiterMap::new())),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        http_client: hyper_util::client::legacy::Client::builder(
            hyper_util::rt::TokioExecutor::new(),
        ).build_http(),
    };

    (state, secret)
}

#[tokio::test]
async fn health_endpoint_needs_no_auth() {
    let (state, _secret) = make_state_with_key().await;
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn api_chat_without_auth_returns_401() {
    let (state, _secret) = make_state_with_key().await;
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/api/chat")
        .method("POST")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn blocked_endpoint_returns_403() {
    let (state, secret) = make_state_with_key().await;
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/api/delete")
        .method("DELETE")
        .header("Authorization", format!("Bearer {secret}"))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
#[ignore] // Requires running Ollama
async fn proxy_to_ollama_tags() {
    let (state, secret) = make_state_with_key().await;
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/api/tags")
        .method("GET")
        .header("Authorization", format!("Bearer {secret}"))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
```

- [ ] **Step 2: Run non-ignored tests**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p gateway --test integration -- --nocapture`

- [ ] **Step 3: Run full test suite with Ollama (manual)**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p gateway -- --nocapture --include-ignored`

---

### Task 9: Make `build_router` and types public for tests

**Files:** `crates/gateway/src/main.rs`

- [ ] **Step 1: Restructure main.rs into lib.rs + main.rs**

The integration tests need access to `build_router` and `AppState`. Axum binary crates typically split into a library and a thin binary entry point.

Create `rust/crates/gateway/src/lib.rs`:

```rust
pub mod auth;
pub mod filter;
pub mod keys;
pub mod proxy;
pub mod ratelimit;
pub mod tls;

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::keys::KeyStore;
use crate::ratelimit::RateLimiterMap;

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub key_store: Arc<RwLock<KeyStore>>,
    pub rate_limiters: Arc<RwLock<RateLimiterMap>>,
    pub ollama_url: String,
    pub http_client: hyper_util::client::legacy::Client<
        hyper_util::client::legacy::connect::HttpConnector,
        axum::body::Body,
    >,
}

/// Build the axum Router with all middleware layers.
pub fn build_router(state: AppState) -> axum::Router {
    use axum::routing::{any, get};

    axum::Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/{*path}", any(proxy::proxy_handler))
        .route(
            "/v1/chat/completions",
            axum::routing::post(proxy::openai_compat_handler),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            ratelimit::rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .with_state(state)
}
```

Update `rust/crates/gateway/src/main.rs` to import from the library:

```rust
use std::net::SocketAddr;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use gateway::{build_router, keys::KeyStore, ratelimit::RateLimiterMap, tls, AppState};

// ... (CLI structs and main() remain the same, but use `gateway::` imports)
```

Also add to `Cargo.toml`:

```toml
[lib]
name = "gateway"
path = "src/lib.rs"
```

- [ ] **Step 2: Verify all tests pass**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p gateway -- --nocapture`

---

### Task 10: Final verification and workspace check

**Files:** None (verification only)

- [ ] **Step 1: Format check**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo fmt -p gateway -- --check`

- [ ] **Step 2: Clippy check**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo clippy -p gateway --all-targets -- -D warnings`

- [ ] **Step 3: Full workspace test**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test --workspace`

- [ ] **Step 4: Smoke test the binary**

Run:
```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
cargo build -p gateway
./target/debug/claw-gateway key create --name "test-key"
./target/debug/claw-gateway key list
./target/debug/claw-gateway serve --tls none --port 9999 &
# In another terminal:
curl -H "Authorization: Bearer <key-from-above>" http://localhost:9999/health
```

- [ ] **Step 5: Test with actual Ollama (manual)**

Start Ollama, then:
```bash
./target/debug/claw-gateway serve --tls none --port 9999 &
curl -H "Authorization: Bearer <key>" http://localhost:9999/api/tags
curl -X POST -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:31b","messages":[{"role":"user","content":"Hello"}]}' \
  http://localhost:9999/api/chat
```
