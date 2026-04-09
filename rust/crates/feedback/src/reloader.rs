use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use crate::finetune_dir;

const SANITY_PROMPTS: &[&str] = &[
    "Write a Python function that reverses a string.",
    "Explain what a mutex is in one sentence.",
    "Fix this code: `fn main() { let x: i32 = \"hello\"; }`",
];

/// Minimum GGUF file size in bytes (~1 GB) -- anything smaller is likely corrupt.
const MIN_GGUF_SIZE: u64 = 1_000_000_000;

/// Maximum model versions to keep on disk.
const DEFAULT_RETENTION: u32 = 3;

/// Error from model reload operations.
#[derive(Debug)]
pub enum ReloadError {
    Io(std::io::Error),
    InvalidGguf(String),
    OllamaCreateFailed(String),
    SanityTestFailed { prompt_index: usize, reason: String },
    RollbackFailed(String),
}

impl std::fmt::Display for ReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::InvalidGguf(msg) => write!(f, "Invalid GGUF: {msg}"),
            Self::OllamaCreateFailed(msg) => write!(f, "ollama create failed: {msg}"),
            Self::SanityTestFailed {
                prompt_index,
                reason,
            } => {
                write!(f, "Sanity test {prompt_index} failed: {reason}")
            }
            Self::RollbackFailed(msg) => write!(f, "Rollback failed: {msg}"),
        }
    }
}

impl std::error::Error for ReloadError {}

impl From<std::io::Error> for ReloadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Handles GGUF validation, `ollama create`, sanity testing, and model hot-swap.
pub struct ModelReloader {
    models_dir: PathBuf,
    ollama_base_url: String,
    model_retention: u32,
}

impl ModelReloader {
    #[must_use]
    pub fn new() -> Self {
        let ollama_base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self {
            models_dir: finetune_dir().join("models"),
            ollama_base_url,
            model_retention: DEFAULT_RETENTION,
        }
    }

    #[must_use]
    pub fn with_retention(mut self, n: u32) -> Self {
        self.model_retention = n;
        self
    }

    /// Determine the next model version number by scanning existing models.
    #[must_use]
    pub fn next_version(&self) -> u32 {
        if !self.models_dir.exists() {
            return 1;
        }
        let mut max_version = 0u32;
        if let Ok(entries) = fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                // Pattern: gemma4-vN.gguf
                if let Some(v_part) = name.strip_prefix("gemma4-v") {
                    if let Some(num_str) = v_part.strip_suffix(".gguf") {
                        if let Ok(n) = num_str.parse::<u32>() {
                            max_version = max_version.max(n);
                        }
                    }
                }
            }
        }
        max_version + 1
    }

    /// Validate a GGUF file: check existence, minimum size, and magic bytes.
    pub fn validate_gguf(&self, path: &Path) -> Result<(), ReloadError> {
        if !path.exists() {
            return Err(ReloadError::InvalidGguf(format!(
                "File not found: {}",
                path.display()
            )));
        }
        let metadata = fs::metadata(path)?;
        if metadata.len() < MIN_GGUF_SIZE {
            return Err(ReloadError::InvalidGguf(format!(
                "File too small: {} bytes (minimum {MIN_GGUF_SIZE})",
                metadata.len(),
            )));
        }
        // Check GGUF magic bytes: 0x47 0x47 0x55 0x46 ("GGUF")
        let header = fs::read(path)
            .map(|data| data.into_iter().take(4).collect::<Vec<_>>())
            .unwrap_or_default();
        if header != b"GGUF" {
            return Err(ReloadError::InvalidGguf(
                "Missing GGUF magic bytes".to_string(),
            ));
        }
        Ok(())
    }

    /// Copy the GGUF into the models directory with a versioned name.
    pub fn install_gguf(&self, source: &Path) -> Result<(PathBuf, String), ReloadError> {
        fs::create_dir_all(&self.models_dir)?;
        let version = self.next_version();
        let model_name = format!("claw-gemma4-v{version}");
        let dest = self.models_dir.join(format!("gemma4-v{version}.gguf"));
        fs::copy(source, &dest)?;
        Ok((dest, model_name))
    }

    /// Create an Ollama model from a GGUF via `ollama create`.
    pub async fn ollama_create(
        &self,
        model_name: &str,
        gguf_path: &Path,
    ) -> Result<(), ReloadError> {
        // Generate a Modelfile.
        let modelfile_path = self.models_dir.join("Modelfile");
        let modelfile_content = format!(
            "FROM {gguf}\nPARAMETER num_ctx 32768\nPARAMETER temperature 0.7\n",
            gguf = gguf_path.display()
        );
        fs::write(&modelfile_path, &modelfile_content)?;

        let output = tokio::process::Command::new("ollama")
            .args(["create", model_name, "-f"])
            .arg(&modelfile_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(ReloadError::OllamaCreateFailed(stderr));
        }
        Ok(())
    }

    /// Run sanity tests against the new model via the Ollama API.
    pub async fn sanity_test(&self, model_name: &str) -> Result<(), ReloadError> {
        let client = reqwest::Client::new();
        for (i, prompt) in SANITY_PROMPTS.iter().enumerate() {
            let body = serde_json::json!({
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": false,
                "options": {"num_predict": 256}
            });
            let resp = client
                .post(format!("{}/api/chat", self.ollama_base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| ReloadError::SanityTestFailed {
                    prompt_index: i,
                    reason: e.to_string(),
                })?;

            if !resp.status().is_success() {
                return Err(ReloadError::SanityTestFailed {
                    prompt_index: i,
                    reason: format!("HTTP {}", resp.status()),
                });
            }

            let json: serde_json::Value =
                resp.json()
                    .await
                    .map_err(|e| ReloadError::SanityTestFailed {
                        prompt_index: i,
                        reason: e.to_string(),
                    })?;

            let content = json
                .pointer("/message/content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if content.len() < 10 {
                return Err(ReloadError::SanityTestFailed {
                    prompt_index: i,
                    reason: format!("Response too short ({} chars)", content.len()),
                });
            }
        }
        Ok(())
    }

    /// Full reload pipeline: validate -> install -> ollama create -> sanity test.
    pub async fn reload(&self, gguf_source: &Path) -> Result<String, ReloadError> {
        self.validate_gguf(gguf_source)?;
        let (installed_path, model_name) = self.install_gguf(gguf_source)?;
        self.ollama_create(&model_name, &installed_path).await?;
        self.sanity_test(&model_name).await?;
        self.prune_old_models();
        Ok(model_name)
    }

    /// List all installed model versions in descending order.
    #[must_use]
    pub fn list_versions(&self) -> Vec<(u32, PathBuf)> {
        let mut versions = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(v_part) = name.strip_prefix("gemma4-v") {
                    if let Some(num_str) = v_part.strip_suffix(".gguf") {
                        if let Ok(n) = num_str.parse::<u32>() {
                            versions.push((n, entry.path()));
                        }
                    }
                }
            }
        }
        versions.sort_by(|a, b| b.0.cmp(&a.0));
        versions
    }

    /// Rollback to the previous model version.
    pub async fn rollback(&self) -> Result<String, ReloadError> {
        let versions = self.list_versions();
        if versions.len() < 2 {
            return Err(ReloadError::RollbackFailed(
                "No previous version to rollback to".to_string(),
            ));
        }
        // versions[0] is current (newest), versions[1] is previous.
        let (prev_version, prev_path) = &versions[1];
        let model_name = format!("claw-gemma4-v{prev_version}");
        self.ollama_create(&model_name, prev_path).await?;
        Ok(model_name)
    }

    /// Remove old model versions, keeping only `model_retention` most recent.
    fn prune_old_models(&self) {
        let versions = self.list_versions();
        if versions.len() <= self.model_retention as usize {
            return;
        }
        for (_, path) in versions.iter().skip(self.model_retention as usize) {
            if let Err(e) = fs::remove_file(path) {
                eprintln!("Warning: failed to prune {}: {e}", path.display());
            }
        }
    }
}

impl Default for ModelReloader {
    fn default() -> Self {
        Self::new()
    }
}
