pub mod exporter;
pub mod logger;
pub mod quality;
pub mod reloader;
pub mod runner;
pub mod trigger;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level feedback pipeline configuration, typically loaded from
/// `~/.claw/config.json` under the `"finetune"` key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    pub auto_enabled: bool,
    pub min_samples: u32,
    pub interval_days: u32,
    pub python_venv: PathBuf,
    pub training: TrainingConfig,
    pub export: ExportConfig,
    pub model_retention: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub max_seq_length: u32,
    pub lora_rank: u32,
    pub lora_alpha: u32,
    pub epochs: u32,
    pub learning_rate: f64,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    pub quantization: String,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            auto_enabled: false,
            min_samples: 100,
            interval_days: 7,
            python_venv: claw_home().join("finetune/venv"),
            training: TrainingConfig::default(),
            export: ExportConfig {
                quantization: "q4_k_m".to_string(),
            },
            model_retention: 3,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_seq_length: 4096,
            lora_rank: 32,
            lora_alpha: 64,
            epochs: 1,
            learning_rate: 2e-4,
            batch_size: 1,
            gradient_accumulation: 8,
        }
    }
}

/// Returns the base Claw home directory (`~/.claw`).
#[must_use]
pub fn claw_home() -> PathBuf {
    dirs_or_fallback().join(".claw")
}

fn dirs_or_fallback() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:/Users/default"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
    }
}

/// Returns the session logs directory (`~/.claw/logs/`).
#[must_use]
pub fn logs_dir() -> PathBuf {
    claw_home().join("logs")
}

/// Returns the fine-tuning data directory (`~/.claw/finetune/`).
#[must_use]
pub fn finetune_dir() -> PathBuf {
    claw_home().join("finetune")
}
