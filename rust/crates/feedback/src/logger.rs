use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// A single log entry for one message or tool event in a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLogEntry {
    pub session_id: String,
    pub timestamp_ms: u64,
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Footer record written when a session ends.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionEndRecord {
    #[serde(rename = "type")]
    record_type: String,
    session_id: String,
    timestamp_ms: u64,
    feedback: String,
}

/// Loaded representation of a completed session log file.
#[derive(Debug, Clone)]
pub struct SessionLog {
    pub session_id: String,
    pub entries: Vec<SessionLogEntry>,
    pub feedback: Option<String>,
}

impl SessionLog {
    /// Load and parse a session JSONL file from disk.
    pub fn load_from_path(path: &Path) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        let mut entries = Vec::new();
        let mut session_id = String::new();
        let mut feedback = None;

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            if value.get("type").and_then(|v| v.as_str()) == Some("session_end") {
                feedback = value
                    .get("feedback")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                if let Some(sid) = value.get("session_id").and_then(|v| v.as_str()) {
                    session_id = sid.to_string();
                }
            } else {
                let entry: SessionLogEntry = serde_json::from_value(value)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                if session_id.is_empty() {
                    session_id.clone_from(&entry.session_id);
                }
                entries.push(entry);
            }
        }

        Ok(Self {
            session_id,
            entries,
            feedback,
        })
    }
}

/// Appends structured log entries to per-session JSONL files.
pub struct SessionLogger {
    log_dir: PathBuf,
}

impl SessionLogger {
    #[must_use]
    pub fn new(log_dir: PathBuf) -> Self {
        Self { log_dir }
    }

    /// Create the log directory if it does not exist.
    pub fn ensure_dir(&self) -> Result<(), std::io::Error> {
        fs::create_dir_all(&self.log_dir)
    }

    /// Append a single entry to the session's JSONL file.
    pub fn append_entry(
        &self,
        session_id: &str,
        entry: &SessionLogEntry,
    ) -> Result<(), std::io::Error> {
        let path = self.session_path(session_id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        let json = serde_json::to_string(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(file, "{json}")
    }

    /// Write a session-end footer with the overall feedback verdict.
    pub fn finalize_session(&self, session_id: &str, feedback: &str) -> Result<(), std::io::Error> {
        let path = self.session_path(session_id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        let record = SessionEndRecord {
            record_type: "session_end".to_string(),
            session_id: session_id.to_string(),
            timestamp_ms: current_time_ms(),
            feedback: feedback.to_string(),
        };
        let json = serde_json::to_string(&record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(file, "{json}")
    }

    /// List all session log files in the log directory.
    pub fn list_sessions(&self) -> Result<Vec<PathBuf>, std::io::Error> {
        let mut paths = Vec::new();
        if !self.log_dir.exists() {
            return Ok(paths);
        }
        for entry in fs::read_dir(&self.log_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.log_dir.join(format!("{session_id}.jsonl"))
    }
}

#[allow(clippy::cast_possible_truncation)]
fn current_time_ms() -> u64 {
    // Truncation from u128 to u64 is safe: u64 millis covers ~584M years.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
