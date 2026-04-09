use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
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
                .filter_map(|k| {
                    k.id.strip_prefix("key_")
                        .and_then(|n| n.parse::<u32>().ok())
                })
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
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&ApiKey> {
        self.keys.iter().find(|k| k.id == id)
    }

    /// Validate a Bearer secret. Returns the key if found AND enabled.
    #[must_use]
    pub fn validate_secret(&self, secret: &str) -> Option<&ApiKey> {
        let hash = hash_secret(secret);
        self.keys
            .iter()
            .find(|k| k.secret_hash == hash && k.enabled)
    }

    /// Print a human-readable table of all keys.
    pub fn print_table(&self) {
        println!(
            "{:<10} {:<20} {:<8} {:<10}",
            "ID", "Name", "Limit", "Status"
        );
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
