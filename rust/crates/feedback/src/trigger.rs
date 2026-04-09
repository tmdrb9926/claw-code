use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::exporter::DataExporter;
use crate::logger::{SessionLog, SessionLogger};
use crate::reloader::ModelReloader;
use crate::runner::FineTuneRunner;
use crate::{finetune_dir, FeedbackConfig};

/// State persisted between runs to track trigger conditions.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TriggerState {
    pub last_finetune_timestamp: Option<u64>,
    pub sessions_since_last_finetune: u32,
    pub auto_enabled: bool,
}

impl TriggerState {
    /// Load state from disk or return default.
    #[must_use]
    pub fn load() -> Self {
        let path = trigger_state_path();
        fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist state to disk.
    pub fn save(&self) -> Result<(), std::io::Error> {
        let path = trigger_state_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(path, json)
    }

    /// Increment the new-session counter and persist.
    pub fn record_session(&mut self) -> Result<(), std::io::Error> {
        self.sessions_since_last_finetune += 1;
        self.save()
    }

    /// Check if automatic fine-tuning should be triggered.
    #[must_use]
    pub fn should_trigger(&self, config: &FeedbackConfig) -> bool {
        if !self.auto_enabled {
            return false;
        }
        if self.sessions_since_last_finetune < config.min_samples {
            return false;
        }
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let interval_secs = u64::from(config.interval_days) * 86_400;
        match self.last_finetune_timestamp {
            Some(ts) => now_secs.saturating_sub(ts) >= interval_secs,
            None => true, // Never run before -- eligible.
        }
    }

    /// Mark a fine-tune as completed, resetting counters.
    pub fn record_finetune_complete(&mut self) -> Result<(), std::io::Error> {
        self.last_finetune_timestamp = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
        self.sessions_since_last_finetune = 0;
        self.save()
    }
}

fn trigger_state_path() -> PathBuf {
    finetune_dir().join("trigger_state.json")
}

/// Status summary for `claw finetune status`.
#[derive(Debug, Clone)]
pub struct PipelineStatus {
    pub auto_enabled: bool,
    pub sessions_since_last: u32,
    pub last_finetune_timestamp: Option<u64>,
    pub total_session_logs: u32,
    pub exportable_sessions: u32,
    pub installed_model_versions: Vec<String>,
}

/// Orchestrator that ties together all pipeline stages.
pub struct PipelineOrchestrator {
    pub config: FeedbackConfig,
    pub logger: SessionLogger,
    pub exporter: DataExporter,
    pub reloader: ModelReloader,
}

impl PipelineOrchestrator {
    #[must_use]
    pub fn new(config: FeedbackConfig) -> Self {
        let logger = SessionLogger::new(crate::logs_dir());
        let exporter = DataExporter::new();
        let reloader = ModelReloader::new().with_retention(config.model_retention);
        Self {
            config,
            logger,
            exporter,
            reloader,
        }
    }

    /// Gather status information for the CLI.
    #[allow(clippy::cast_possible_truncation)]
    pub fn status(&self) -> Result<PipelineStatus, std::io::Error> {
        let trigger_state = TriggerState::load();
        let session_paths = self.logger.list_sessions()?;
        let total_session_logs = session_paths.len() as u32;

        let logs = self.load_all_session_logs()?;
        let export_stats = self.exporter.compute_stats(&logs);
        let versions = self.reloader.list_versions();
        let version_names: Vec<String> = versions
            .iter()
            .map(|(v, _)| format!("claw-gemma4-v{v}"))
            .collect();

        Ok(PipelineStatus {
            auto_enabled: trigger_state.auto_enabled,
            sessions_since_last: trigger_state.sessions_since_last_finetune,
            last_finetune_timestamp: trigger_state.last_finetune_timestamp,
            total_session_logs,
            exportable_sessions: export_stats.sessions_exported,
            installed_model_versions: version_names,
        })
    }

    /// Export training data from all session logs.
    pub fn export_data(&self) -> Result<crate::exporter::ExportStats, std::io::Error> {
        let logs = self.load_all_session_logs()?;
        let output_path = finetune_dir().join("data/train.jsonl");
        self.exporter.export_to_jsonl(&logs, &output_path)
    }

    /// Run the full fine-tuning pipeline: export -> train -> reload.
    pub async fn run_finetune(&self) -> Result<String, Box<dyn std::error::Error>> {
        // Step 1: Export data.
        let export_result = self.export_data()?;
        if export_result.conversations_written == 0 {
            return Err("No training data available".into());
        }
        let data_path = finetune_dir().join("data/train.jsonl");

        // Step 2: Run fine-tuning.
        let runner = FineTuneRunner::new(
            "unsloth/gemma-3-27b-it-bnb-4bit",
            self.config.training.clone(),
            &self.config.export.quantization,
            self.config.python_venv.clone(),
        );
        let result = runner.run(&data_path).await?;

        // Step 3: Reload model.
        let model_name = self
            .reloader
            .reload(&result.gguf_path)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Step 4: Update trigger state.
        let mut trigger_state = TriggerState::load();
        trigger_state.record_finetune_complete()?;

        Ok(model_name)
    }

    /// Rollback to the previous model version.
    pub async fn rollback(&self) -> Result<String, Box<dyn std::error::Error>> {
        let model_name = self
            .reloader
            .rollback()
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        Ok(model_name)
    }

    /// Enable or disable automatic fine-tuning.
    pub fn set_auto(&self, enabled: bool) -> Result<(), std::io::Error> {
        let mut trigger_state = TriggerState::load();
        trigger_state.auto_enabled = enabled;
        trigger_state.save()
    }

    /// Load all session logs from the log directory.
    fn load_all_session_logs(&self) -> Result<Vec<SessionLog>, std::io::Error> {
        let paths = self.logger.list_sessions()?;
        let mut logs = Vec::new();
        for path in paths {
            match SessionLog::load_from_path(&path) {
                Ok(log) => logs.push(log),
                Err(e) => {
                    eprintln!("Warning: skipping {}: {e}", path.display());
                }
            }
        }
        Ok(logs)
    }
}
