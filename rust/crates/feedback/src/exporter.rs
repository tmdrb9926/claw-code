use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::logger::SessionLog;

/// One message in the Unsloth/ShareGPT training format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMessage {
    pub role: String,
    pub content: String,
}

/// One training example: a multi-turn conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConversation {
    pub conversations: Vec<TrainingMessage>,
}

/// Statistics returned after an export operation.
#[derive(Debug, Clone, Copy)]
pub struct ExportStats {
    pub sessions_exported: u32,
    pub sessions_skipped: u32,
    pub conversations_written: u32,
}

/// Converts session logs into fine-tuning JSONL format.
#[derive(Debug, Clone)]
pub struct DataExporter {
    /// Maximum token-length estimate (character-based heuristic) per conversation.
    max_char_length: usize,
}

impl DataExporter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            // ~4096 tokens * 4 chars/token average = 16384 chars
            max_char_length: 16_384,
        }
    }

    /// Convert a single session log into zero or more training conversations.
    ///
    /// Returns an empty Vec if the session has no accepted feedback or contains
    /// only tool-result messages.
    #[must_use]
    pub fn convert_session(&self, log: &SessionLog) -> Vec<TrainingConversation> {
        // Only export sessions with positive/accepted feedback.
        match log.feedback.as_deref() {
            Some("accepted") => {}
            _ => return Vec::new(),
        }

        let messages: Vec<TrainingMessage> = log
            .entries
            .iter()
            .filter(|entry| entry.role != "tool") // Remove tool_result entries
            .map(|entry| TrainingMessage {
                role: entry.role.clone(),
                content: entry.content.clone(),
            })
            .collect();

        if messages.len() < 2 {
            return Vec::new();
        }

        // Simple heuristic: total character length must be under threshold.
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        if total_chars > self.max_char_length {
            return Vec::new();
        }

        vec![TrainingConversation {
            conversations: messages,
        }]
    }

    /// Export multiple session logs to a JSONL file, returning stats.
    pub fn export_to_jsonl(
        &self,
        logs: &[SessionLog],
        output_path: &Path,
    ) -> Result<ExportStats, std::io::Error> {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        let mut sessions_exported = 0u32;
        let mut sessions_skipped = 0u32;
        let mut conversations_written = 0u32;

        for log in logs {
            let conversations = self.convert_session(log);
            if conversations.is_empty() {
                sessions_skipped += 1;
                continue;
            }
            sessions_exported += 1;
            for conv in &conversations {
                let json = serde_json::to_string(conv)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                writeln!(writer, "{json}")?;
                conversations_written += 1;
            }
        }

        writer.flush()?;
        Ok(ExportStats {
            sessions_exported,
            sessions_skipped,
            conversations_written,
        })
    }

    /// Compute statistics about available session logs without exporting.
    #[must_use]
    pub fn compute_stats(&self, logs: &[SessionLog]) -> ExportStats {
        let mut sessions_exported = 0u32;
        let mut sessions_skipped = 0u32;
        let mut conversations_written = 0u32;

        for log in logs {
            let conversations = self.convert_session(log);
            if conversations.is_empty() {
                sessions_skipped += 1;
            } else {
                sessions_exported += 1;
                conversations_written += conversations.len() as u32;
            }
        }

        ExportStats {
            sessions_exported,
            sessions_skipped,
            conversations_written,
        }
    }
}

impl Default for DataExporter {
    fn default() -> Self {
        Self::new()
    }
}
