use std::net::SocketAddr;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use gateway::{build_router, keys::KeyStore, ratelimit::RateLimiterMap, tls, AppState};

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
        /// Key ID to revoke (e.g. "`key_01`")
        id: String,
    },
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

            let http_client =
                hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
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
                    let cert_path = cert.expect("--cert is required when --tls=files");
                    let key_path = key_file.expect("--key-file is required when --tls=files");
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

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!("Shutdown signal received");
}
