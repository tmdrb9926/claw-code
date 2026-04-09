# FeedbackPipeline Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-improving feedback loop that automatically collects session data from Claw Code conversations, converts positive-signal sessions into fine-tuning JSONL, runs Unsloth QLoRA fine-tuning via subprocess, and hot-swaps the resulting GGUF model into Ollama.

**Architecture:** A new `feedback` crate with six modules (logger, exporter, runner, reloader, trigger, quality) integrates into the existing runtime via a `SessionLogger` hook called after each turn. The pipeline flows: SessionLogger writes JSONL session logs to `~/.claw/logs/`, DataExporter filters and converts them to training JSONL, FineTuneRunner spawns a Python subprocess (auto-generated Unsloth script), and ModelReloader creates Ollama models from the output GGUF and does sanity testing before hot-swap. CLI commands under `claw finetune` expose manual and auto triggers.

**Tech Stack:** Rust (serde, serde_json, tokio::process, chrono), Python (Unsloth, auto-generated script), Ollama CLI (`ollama create`/`ollama run`)

**Phases overview:**
- **Phase 1 (done):** OllamaClient provider -- local Gemma 4 inference via Claw Code
- **Phase 2 (this plan):** FeedbackPipeline -- session logging + fine-tuning loop
- **Phase 3 (separate plan):** API Gateway -- external access with auth + rate limiting

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `crates/feedback/Cargo.toml` | Crate manifest with serde, serde_json, tokio, chrono dependencies |
| Create | `crates/feedback/src/lib.rs` | Public module declarations + re-exports + `FeedbackConfig` |
| Create | `crates/feedback/src/logger.rs` | `SessionLogger` -- hooks into runtime, writes session JSONL to `~/.claw/logs/` |
| Create | `crates/feedback/src/quality.rs` | `QualityAnalyzer` -- positive/negative signal detection from messages |
| Create | `crates/feedback/src/exporter.rs` | `DataExporter` -- filters sessions by quality, converts to training JSONL |
| Create | `crates/feedback/src/runner.rs` | `FineTuneRunner` -- generates Python script, spawns Unsloth subprocess |
| Create | `crates/feedback/src/reloader.rs` | `ModelReloader` -- GGUF validation, `ollama create`, sanity test, hot-swap |
| Create | `crates/feedback/src/trigger.rs` | `AutoTrigger` -- checks data count + time thresholds, orchestrates full pipeline |
| Create | `crates/feedback/tests/quality_tests.rs` | Unit tests for quality signal detection |
| Create | `crates/feedback/tests/exporter_tests.rs` | Unit tests for session-to-JSONL conversion |
| Create | `crates/feedback/tests/logger_tests.rs` | Unit tests for session log writing |
| Modify | `crates/runtime/Cargo.toml` | Add `feedback = { path = "../feedback" }` dependency |
| Modify | `crates/runtime/src/conversation.rs` | Call `SessionLogger::on_turn_complete()` after each turn |
| Modify | `crates/runtime/src/lib.rs` | Re-export feedback types needed by CLI |
| Modify | `crates/commands/src/lib.rs` | Add `finetune` slash command specs |
| Modify | `rust/Cargo.toml` | Workspace already auto-discovers `crates/*` -- no change needed |

---

### Task 1: Create feedback crate scaffold

**Files:**
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/Cargo.toml`
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/src/lib.rs`

- [ ] **Step 1: Create `Cargo.toml` for the feedback crate**

Create `crates/feedback/Cargo.toml`:

```toml
[package]
name = "feedback"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
chrono = { version = "0.4", features = ["serde"] }
runtime = { path = "../runtime" }
serde = { version = "1", features = ["derive"] }
serde_json.workspace = true
tokio = { version = "1", features = ["fs", "io-util", "macros", "process", "rt", "time"] }

[dev-dependencies]
tempfile = "3"

[lints]
workspace = true
```

- [ ] **Step 2: Create `lib.rs` with module declarations, config struct, and re-exports**

Create `crates/feedback/src/lib.rs`:

```rust
pub mod exporter;
pub mod logger;
pub mod quality;
pub mod reloader;
pub mod runner;
pub mod trigger;

use std::path::{Path, PathBuf};

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
```

- [ ] **Step 3: Verify crate compiles**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p feedback`

---

### Task 2: Implement quality.rs -- feedback signal detection

**Files:**
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/src/quality.rs`
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/tests/quality_tests.rs`

- [ ] **Step 1: Write failing tests for quality analysis**

Create `crates/feedback/tests/quality_tests.rs`:

```rust
use feedback::quality::{FeedbackSignal, QualityAnalyzer};

#[test]
fn positive_explicit_approval() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("좋아, 잘 동작해");
    assert_eq!(signal, FeedbackSignal::Positive);
}

#[test]
fn positive_english_approval() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("perfect, that works great");
    assert_eq!(signal, FeedbackSignal::Positive);
}

#[test]
fn negative_retry_request() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("아니, 다시 해줘");
    assert_eq!(signal, FeedbackSignal::Negative);
}

#[test]
fn negative_english_rejection() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("no, that's wrong, try again");
    assert_eq!(signal, FeedbackSignal::Negative);
}

#[test]
fn neutral_followup_question() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("이제 테스트 코드도 작성해줘");
    assert_eq!(signal, FeedbackSignal::Neutral);
}

#[test]
fn tool_success_is_low_positive() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_tool_result(/* is_error */ false);
    assert_eq!(signal, FeedbackSignal::WeakPositive);
}

#[test]
fn tool_error_is_weak_negative() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_tool_result(/* is_error */ true);
    assert_eq!(signal, FeedbackSignal::WeakNegative);
}

#[test]
fn session_overall_quality_positive() {
    let analyzer = QualityAnalyzer::new();
    let signals = vec![
        FeedbackSignal::WeakPositive,
        FeedbackSignal::WeakPositive,
        FeedbackSignal::Positive,
    ];
    assert!(analyzer.is_session_positive(&signals));
}

#[test]
fn session_with_negative_not_positive() {
    let analyzer = QualityAnalyzer::new();
    let signals = vec![
        FeedbackSignal::WeakPositive,
        FeedbackSignal::Negative,
        FeedbackSignal::Neutral,
    ];
    assert!(!analyzer.is_session_positive(&signals));
}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p feedback --test quality_tests -- --nocapture`

- [ ] **Step 3: Implement `QualityAnalyzer` with signal detection**

Create `crates/feedback/src/quality.rs`:

```rust
use serde::{Deserialize, Serialize};

/// Strength and direction of a feedback signal extracted from a single event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackSignal {
    /// Explicit positive language ("좋아", "perfect", "thanks").
    Positive,
    /// Tool executed successfully without error.
    WeakPositive,
    /// No clear signal -- follow-up question or neutral statement.
    Neutral,
    /// Tool execution failed.
    WeakNegative,
    /// Explicit negative language ("아니", "wrong", "다시").
    Negative,
}

impl FeedbackSignal {
    /// Numeric weight used when aggregating signals for a full session.
    #[must_use]
    pub fn weight(self) -> i32 {
        match self {
            Self::Positive => 3,
            Self::WeakPositive => 1,
            Self::Neutral => 0,
            Self::WeakNegative => -1,
            Self::Negative => -3,
        }
    }
}

/// Analyzes conversation events for implicit and explicit feedback signals.
#[derive(Debug, Clone)]
pub struct QualityAnalyzer {
    positive_patterns: Vec<&'static str>,
    negative_patterns: Vec<&'static str>,
}

impl QualityAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            positive_patterns: vec![
                // Korean
                "좋아", "완벽", "잘 동작", "고마워", "감사", "맞아", "좋네", "훌륭",
                // English
                "perfect", "great", "thanks", "works", "good", "excellent", "nice",
                "awesome", "correct", "exactly",
            ],
            negative_patterns: vec![
                // Korean
                "아니", "다시", "틀렸", "잘못", "아닌데", "안 돼", "에러", "고쳐",
                // English
                "wrong", "no,", "try again", "incorrect", "fix", "broken", "doesn't work",
                "not right", "redo",
            ],
        }
    }

    /// Classify a user message as positive, negative, or neutral.
    #[must_use]
    pub fn analyze_user_message(&self, text: &str) -> FeedbackSignal {
        let lower = text.to_lowercase();

        // Check negative first -- explicit rejection is a strong signal.
        for pattern in &self.negative_patterns {
            if lower.contains(pattern) {
                return FeedbackSignal::Negative;
            }
        }

        for pattern in &self.positive_patterns {
            if lower.contains(pattern) {
                return FeedbackSignal::Positive;
            }
        }

        FeedbackSignal::Neutral
    }

    /// Classify a tool execution result.
    #[must_use]
    pub fn analyze_tool_result(&self, is_error: bool) -> FeedbackSignal {
        if is_error {
            FeedbackSignal::WeakNegative
        } else {
            FeedbackSignal::WeakPositive
        }
    }

    /// Determine if a session's aggregated signals indicate positive quality.
    ///
    /// A session is considered positive if:
    /// 1. No explicit `Negative` signal exists, AND
    /// 2. The weighted sum of all signals is strictly positive.
    #[must_use]
    pub fn is_session_positive(&self, signals: &[FeedbackSignal]) -> bool {
        let has_negative = signals.iter().any(|s| *s == FeedbackSignal::Negative);
        if has_negative {
            return false;
        }
        let total_weight: i32 = signals.iter().map(|s| s.weight()).sum();
        total_weight > 0
    }
}

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p feedback --test quality_tests -- --nocapture`

---

### Task 3: Implement logger.rs -- session log writer

**Files:**
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/src/logger.rs`
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/tests/logger_tests.rs`

- [ ] **Step 1: Write failing tests for session logging**

Create `crates/feedback/tests/logger_tests.rs`:

```rust
use std::fs;
use tempfile::TempDir;

use feedback::logger::{SessionLog, SessionLogEntry, SessionLogger};

#[test]
fn logger_creates_log_directory() {
    let tmp = TempDir::new().unwrap();
    let log_dir = tmp.path().join("logs");
    let logger = SessionLogger::new(log_dir.clone());
    logger.ensure_dir().unwrap();
    assert!(log_dir.exists());
}

#[test]
fn logger_writes_jsonl_entry() {
    let tmp = TempDir::new().unwrap();
    let log_dir = tmp.path().join("logs");
    let logger = SessionLogger::new(log_dir.clone());
    logger.ensure_dir().unwrap();

    let entry = SessionLogEntry {
        session_id: "sess_001".to_string(),
        timestamp_ms: 1_000_000,
        role: "user".to_string(),
        content: "hello world".to_string(),
        tool_name: None,
        tool_input: None,
        is_error: None,
    };
    logger.append_entry("sess_001", &entry).unwrap();

    let log_path = log_dir.join("sess_001.jsonl");
    assert!(log_path.exists());
    let contents = fs::read_to_string(&log_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(contents.trim()).unwrap();
    assert_eq!(parsed["role"], "user");
    assert_eq!(parsed["content"], "hello world");
}

#[test]
fn logger_finalizes_session_with_feedback() {
    let tmp = TempDir::new().unwrap();
    let log_dir = tmp.path().join("logs");
    let logger = SessionLogger::new(log_dir.clone());
    logger.ensure_dir().unwrap();

    let entry = SessionLogEntry {
        session_id: "sess_002".to_string(),
        timestamp_ms: 2_000_000,
        role: "user".to_string(),
        content: "test".to_string(),
        tool_name: None,
        tool_input: None,
        is_error: None,
    };
    logger.append_entry("sess_002", &entry).unwrap();
    logger.finalize_session("sess_002", "accepted").unwrap();

    let log_path = log_dir.join("sess_002.jsonl");
    let contents = fs::read_to_string(&log_path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert!(lines.len() >= 2);
    let footer: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(footer["type"], "session_end");
    assert_eq!(footer["feedback"], "accepted");
}

#[test]
fn load_session_log_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let log_dir = tmp.path().join("logs");
    let logger = SessionLogger::new(log_dir.clone());
    logger.ensure_dir().unwrap();

    let entry = SessionLogEntry {
        session_id: "sess_003".to_string(),
        timestamp_ms: 3_000_000,
        role: "assistant".to_string(),
        content: "I wrote a function".to_string(),
        tool_name: Some("Write".to_string()),
        tool_input: Some(r#"{"path":"foo.rs"}"#.to_string()),
        is_error: None,
    };
    logger.append_entry("sess_003", &entry).unwrap();
    logger.finalize_session("sess_003", "accepted").unwrap();

    let log = SessionLog::load_from_path(&log_dir.join("sess_003.jsonl")).unwrap();
    assert_eq!(log.session_id, "sess_003");
    assert_eq!(log.entries.len(), 1);
    assert_eq!(log.feedback.as_deref(), Some("accepted"));
}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p feedback --test logger_tests -- --nocapture`

- [ ] **Step 3: Implement `SessionLogger`**

Create `crates/feedback/src/logger.rs`:

```rust
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
                feedback = value.get("feedback").and_then(|v| v.as_str()).map(String::from);
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
    pub fn finalize_session(
        &self,
        session_id: &str,
        feedback: &str,
    ) -> Result<(), std::io::Error> {
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

fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p feedback --test logger_tests -- --nocapture`

---

### Task 4: Implement exporter.rs -- session-to-training-JSONL converter

**Files:**
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/src/exporter.rs`
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/tests/exporter_tests.rs`

- [ ] **Step 1: Write failing tests for data export**

Create `crates/feedback/tests/exporter_tests.rs`:

```rust
use std::fs;
use tempfile::TempDir;

use feedback::exporter::{DataExporter, TrainingConversation, TrainingMessage};
use feedback::logger::{SessionLog, SessionLogEntry};

#[test]
fn converts_user_assistant_pair() {
    let log = SessionLog {
        session_id: "sess_001".to_string(),
        entries: vec![
            SessionLogEntry {
                session_id: "sess_001".to_string(),
                timestamp_ms: 1000,
                role: "user".to_string(),
                content: "Implement quicksort".to_string(),
                tool_name: None,
                tool_input: None,
                is_error: None,
            },
            SessionLogEntry {
                session_id: "sess_001".to_string(),
                timestamp_ms: 2000,
                role: "assistant".to_string(),
                content: "```python\ndef quicksort(arr):\n    ...\n```".to_string(),
                tool_name: None,
                tool_input: None,
                is_error: None,
            },
        ],
        feedback: Some("accepted".to_string()),
    };

    let exporter = DataExporter::new();
    let conversations = exporter.convert_session(&log);
    assert_eq!(conversations.len(), 1);
    assert_eq!(conversations[0].conversations.len(), 2);
    assert_eq!(conversations[0].conversations[0].role, "user");
    assert_eq!(conversations[0].conversations[1].role, "assistant");
}

#[test]
fn filters_tool_result_entries() {
    let log = SessionLog {
        session_id: "sess_002".to_string(),
        entries: vec![
            SessionLogEntry {
                session_id: "sess_002".to_string(),
                timestamp_ms: 1000,
                role: "user".to_string(),
                content: "Write a file".to_string(),
                tool_name: None,
                tool_input: None,
                is_error: None,
            },
            SessionLogEntry {
                session_id: "sess_002".to_string(),
                timestamp_ms: 2000,
                role: "assistant".to_string(),
                content: "I'll write the file now.".to_string(),
                tool_name: Some("Write".to_string()),
                tool_input: Some(r#"{"path":"foo.rs"}"#.to_string()),
                is_error: None,
            },
            SessionLogEntry {
                session_id: "sess_002".to_string(),
                timestamp_ms: 3000,
                role: "tool".to_string(),
                content: "File written successfully".to_string(),
                tool_name: Some("Write".to_string()),
                tool_input: None,
                is_error: Some(false),
            },
        ],
        feedback: Some("accepted".to_string()),
    };

    let exporter = DataExporter::new();
    let conversations = exporter.convert_session(&log);
    // tool results should be filtered out
    assert_eq!(conversations[0].conversations.len(), 2);
}

#[test]
fn export_writes_jsonl_file() {
    let tmp = TempDir::new().unwrap();
    let output_path = tmp.path().join("train.jsonl");

    let log = SessionLog {
        session_id: "sess_003".to_string(),
        entries: vec![
            SessionLogEntry {
                session_id: "sess_003".to_string(),
                timestamp_ms: 1000,
                role: "user".to_string(),
                content: "Hello".to_string(),
                tool_name: None,
                tool_input: None,
                is_error: None,
            },
            SessionLogEntry {
                session_id: "sess_003".to_string(),
                timestamp_ms: 2000,
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
                tool_name: None,
                tool_input: None,
                is_error: None,
            },
        ],
        feedback: Some("accepted".to_string()),
    };

    let exporter = DataExporter::new();
    exporter.export_to_jsonl(&[log], &output_path).unwrap();

    assert!(output_path.exists());
    let contents = fs::read_to_string(&output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(contents.trim()).unwrap();
    assert!(parsed["conversations"].is_array());
}

#[test]
fn skips_sessions_without_accepted_feedback() {
    let log_rejected = SessionLog {
        session_id: "sess_bad".to_string(),
        entries: vec![SessionLogEntry {
            session_id: "sess_bad".to_string(),
            timestamp_ms: 1000,
            role: "user".to_string(),
            content: "test".to_string(),
            tool_name: None,
            tool_input: None,
            is_error: None,
        }],
        feedback: Some("rejected".to_string()),
    };

    let exporter = DataExporter::new();
    let conversations = exporter.convert_session(&log_rejected);
    assert!(conversations.is_empty());
}

#[test]
fn data_stats_counts_sessions() {
    let tmp = TempDir::new().unwrap();
    let output = tmp.path().join("train.jsonl");

    let logs = vec![
        SessionLog {
            session_id: "s1".to_string(),
            entries: vec![
                SessionLogEntry { session_id: "s1".to_string(), timestamp_ms: 1, role: "user".to_string(), content: "a".to_string(), tool_name: None, tool_input: None, is_error: None },
                SessionLogEntry { session_id: "s1".to_string(), timestamp_ms: 2, role: "assistant".to_string(), content: "b".to_string(), tool_name: None, tool_input: None, is_error: None },
            ],
            feedback: Some("accepted".to_string()),
        },
        SessionLog {
            session_id: "s2".to_string(),
            entries: vec![
                SessionLogEntry { session_id: "s2".to_string(), timestamp_ms: 3, role: "user".to_string(), content: "c".to_string(), tool_name: None, tool_input: None, is_error: None },
                SessionLogEntry { session_id: "s2".to_string(), timestamp_ms: 4, role: "assistant".to_string(), content: "d".to_string(), tool_name: None, tool_input: None, is_error: None },
            ],
            feedback: Some("accepted".to_string()),
        },
    ];

    let exporter = DataExporter::new();
    let stats = exporter.export_to_jsonl(&logs, &output).unwrap();
    assert_eq!(stats.sessions_exported, 2);
    assert_eq!(stats.conversations_written, 2);
}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p feedback --test exporter_tests -- --nocapture`

- [ ] **Step 3: Implement `DataExporter`**

Create `crates/feedback/src/exporter.rs`:

```rust
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p feedback --test exporter_tests -- --nocapture`

---

### Task 5: Implement runner.rs -- Unsloth fine-tune subprocess

**Files:**
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/src/runner.rs`

- [ ] **Step 1: Implement `FineTuneRunner` with Python script generation and subprocess execution**

Create `crates/feedback/src/runner.rs`:

```rust
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{finetune_dir, TrainingConfig};

/// Result of a fine-tuning run.
#[derive(Debug, Clone)]
pub struct FineTuneResult {
    pub run_dir: PathBuf,
    pub gguf_path: PathBuf,
    pub duration_secs: u64,
    pub success: bool,
    pub log_output: String,
}

/// Error from a fine-tune run.
#[derive(Debug)]
pub enum FineTuneError {
    Io(std::io::Error),
    PythonNotFound,
    ScriptFailed { exit_code: Option<i32>, stderr: String },
    GgufNotFound(PathBuf),
}

impl std::fmt::Display for FineTuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::PythonNotFound => write!(f, "Python not found in PATH or venv"),
            Self::ScriptFailed { exit_code, stderr } => {
                write!(f, "Script failed (exit code {exit_code:?}): {stderr}")
            }
            Self::GgufNotFound(path) => write!(f, "Expected GGUF not found at {}", path.display()),
        }
    }
}

impl std::error::Error for FineTuneError {}

impl From<std::io::Error> for FineTuneError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Manages fine-tuning runs by generating Unsloth Python scripts and executing
/// them as subprocesses.
pub struct FineTuneRunner {
    base_model: String,
    config: TrainingConfig,
    quantization: String,
    venv_path: PathBuf,
}

impl FineTuneRunner {
    #[must_use]
    pub fn new(
        base_model: impl Into<String>,
        config: TrainingConfig,
        quantization: impl Into<String>,
        venv_path: PathBuf,
    ) -> Self {
        Self {
            base_model: base_model.into(),
            config,
            quantization: quantization.into(),
            venv_path,
        }
    }

    /// Generate the Unsloth fine-tuning Python script and write it to the
    /// scripts directory. Returns the path to the generated script.
    pub fn generate_script(
        &self,
        data_path: &Path,
        output_dir: &Path,
    ) -> Result<PathBuf, FineTuneError> {
        let scripts_dir = finetune_dir().join("scripts");
        fs::create_dir_all(&scripts_dir)?;
        let script_path = scripts_dir.join("finetune.py");

        let script = format!(
            r#"#!/usr/bin/env python3
"""Auto-generated Unsloth QLoRA fine-tuning script.
DO NOT EDIT -- regenerated by Claw Code on each run.
"""
import os
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────
BASE_MODEL = "{base_model}"
DATA_PATH = r"{data_path}"
OUTPUT_DIR = r"{output_dir}"
MAX_SEQ_LENGTH = {max_seq_length}
LORA_RANK = {lora_rank}
LORA_ALPHA = {lora_alpha}
EPOCHS = {epochs}
LEARNING_RATE = {learning_rate}
BATCH_SIZE = {batch_size}
GRADIENT_ACCUMULATION = {gradient_accumulation}
QUANTIZATION = "{quantization}"

# ── Load Model ─────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# ── Apply LoRA ─────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ── Load Dataset ───────────────────────────────────────────
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def formatting_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        text = ""
        for msg in convo:
            role = msg["role"]
            content = msg["content"]
            text += f"<start_of_turn>{{role}}\n{{content}}<end_of_turn>\n"
        texts.append(text)
    return {{"text": texts}}

dataset = dataset.map(formatting_func, batched=True)

# ── Train ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="epoch",
        seed=42,
    ),
)

print("Starting training...")
trainer.train()
print("Training complete.")

# ── Export to GGUF ─────────────────────────────────────────
print(f"Exporting to GGUF ({{QUANTIZATION}})...")
model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method=QUANTIZATION,
)
print("GGUF export complete.")

# Write completion marker
with open(os.path.join(OUTPUT_DIR, "DONE"), "w") as f:
    f.write("ok")
"#,
            base_model = self.base_model,
            data_path = data_path.display(),
            output_dir = output_dir.display(),
            max_seq_length = self.config.max_seq_length,
            lora_rank = self.config.lora_rank,
            lora_alpha = self.config.lora_alpha,
            epochs = self.config.epochs,
            learning_rate = self.config.learning_rate,
            batch_size = self.config.batch_size,
            gradient_accumulation = self.config.gradient_accumulation,
            quantization = self.quantization,
        );

        let mut file = File::create(&script_path)?;
        file.write_all(script.as_bytes())?;
        Ok(script_path)
    }

    /// Resolve the Python executable path: prefer the venv, fall back to PATH.
    #[must_use]
    pub fn python_path(&self) -> PathBuf {
        let venv_python = if cfg!(windows) {
            self.venv_path.join("Scripts/python.exe")
        } else {
            self.venv_path.join("bin/python")
        };
        if venv_python.exists() {
            venv_python
        } else {
            PathBuf::from("python3")
        }
    }

    /// Execute the fine-tuning script as a subprocess and wait for completion.
    pub async fn run(
        &self,
        data_path: &Path,
    ) -> Result<FineTuneResult, FineTuneError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let run_dir = finetune_dir().join(format!("runs/run_{timestamp}"));
        fs::create_dir_all(&run_dir)?;

        // Write training config.
        let config_json = serde_json::json!({
            "base_model": self.base_model,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gradient_accumulation": self.config.gradient_accumulation,
            "max_seq_length": self.config.max_seq_length,
            "quantization": self.quantization,
        });
        fs::write(
            run_dir.join("config.json"),
            serde_json::to_string_pretty(&config_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
        )?;

        let output_model_dir = run_dir.join("output");
        let script_path = self.generate_script(data_path, &output_model_dir)?;
        let python = self.python_path();

        let start = std::time::Instant::now();

        let child = tokio::process::Command::new(&python)
            .arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|_| FineTuneError::PythonNotFound)?;

        let output = child.wait_with_output().await?;
        let duration_secs = start.elapsed().as_secs();

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // Save log.
        fs::write(run_dir.join("train.log"), format!("{stdout}\n---STDERR---\n{stderr}"))?;

        if !output.status.success() {
            return Err(FineTuneError::ScriptFailed {
                exit_code: output.status.code(),
                stderr,
            });
        }

        // Find the GGUF file.
        let gguf_path = find_gguf_in_dir(&output_model_dir)?;

        Ok(FineTuneResult {
            run_dir,
            gguf_path,
            duration_secs,
            success: true,
            log_output: stdout,
        })
    }
}

/// Scan a directory for the first `.gguf` file.
fn find_gguf_in_dir(dir: &Path) -> Result<PathBuf, FineTuneError> {
    if !dir.exists() {
        return Err(FineTuneError::GgufNotFound(dir.to_path_buf()));
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(path);
        }
    }
    Err(FineTuneError::GgufNotFound(dir.to_path_buf()))
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p feedback`

---

### Task 6: Implement reloader.rs -- GGUF validation and Ollama model hot-swap

**Files:**
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/src/reloader.rs`

- [ ] **Step 1: Implement `ModelReloader` with GGUF validation, `ollama create`, sanity test, and rollback**

Create `crates/feedback/src/reloader.rs`:

```rust
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
            Self::SanityTestFailed { prompt_index, reason } => {
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
                "File too small: {} bytes (minimum {})",
                metadata.len(),
                MIN_GGUF_SIZE
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

            let json: serde_json::Value = resp
                .json()
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
    pub async fn reload(
        &self,
        gguf_source: &Path,
    ) -> Result<String, ReloadError> {
        self.validate_gguf(gguf_source)?;
        let (installed_path, model_name) = self.install_gguf(gguf_source)?;
        self.ollama_create(&model_name, &installed_path).await?;
        self.sanity_test(&model_name).await?;
        self.prune_old_models()?;
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
    fn prune_old_models(&self) -> Result<(), ReloadError> {
        let versions = self.list_versions();
        if versions.len() as u32 <= self.model_retention {
            return Ok(());
        }
        for (_, path) in versions.iter().skip(self.model_retention as usize) {
            if let Err(e) = fs::remove_file(path) {
                eprintln!("Warning: failed to prune {}: {e}", path.display());
            }
        }
        Ok(())
    }
}

impl Default for ModelReloader {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p feedback`

---

### Task 7: Implement trigger.rs -- auto/manual trigger orchestration

**Files:**
- Create: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/src/trigger.rs`

- [ ] **Step 1: Implement `AutoTrigger` with threshold checking and full pipeline orchestration**

Create `crates/feedback/src/trigger.rs`:

```rust
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::exporter::DataExporter;
use crate::logger::{SessionLog, SessionLogger};
use crate::reloader::ModelReloader;
use crate::runner::FineTuneRunner;
use crate::{finetune_dir, FeedbackConfig};

/// State persisted between runs to track trigger conditions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TriggerState {
    pub last_finetune_timestamp: Option<u64>,
    pub sessions_since_last_finetune: u32,
    pub auto_enabled: bool,
}

impl Default for TriggerState {
    fn default() -> Self {
        Self {
            last_finetune_timestamp: None,
            sessions_since_last_finetune: 0,
            auto_enabled: false,
        }
    }
}

impl TriggerState {
    /// Load state from disk or return default.
    #[must_use]
    pub fn load() -> Self {
        let path = state_path();
        fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist state to disk.
    pub fn save(&self) -> Result<(), std::io::Error> {
        let path = state_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
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

fn state_path() -> PathBuf {
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
    pub fn status(&self) -> Result<PipelineStatus, std::io::Error> {
        let state = TriggerState::load();
        let session_paths = self.logger.list_sessions()?;
        let total_session_logs = session_paths.len() as u32;

        let logs = self.load_all_session_logs()?;
        let stats = self.exporter.compute_stats(&logs);
        let versions = self.reloader.list_versions();
        let version_names: Vec<String> = versions
            .iter()
            .map(|(v, _)| format!("claw-gemma4-v{v}"))
            .collect();

        Ok(PipelineStatus {
            auto_enabled: state.auto_enabled,
            sessions_since_last: state.sessions_since_last_finetune,
            last_finetune_timestamp: state.last_finetune_timestamp,
            total_session_logs,
            exportable_sessions: stats.sessions_exported,
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
        let stats = self.export_data()?;
        if stats.conversations_written == 0 {
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
        let model_name = self.reloader.reload(&result.gguf_path).await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Step 4: Update trigger state.
        let mut state = TriggerState::load();
        state.record_finetune_complete()?;

        Ok(model_name)
    }

    /// Rollback to the previous model version.
    pub async fn rollback(&self) -> Result<String, Box<dyn std::error::Error>> {
        let model_name = self.reloader.rollback().await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        Ok(model_name)
    }

    /// Enable or disable automatic fine-tuning.
    pub fn set_auto(&self, enabled: bool) -> Result<(), std::io::Error> {
        let mut state = TriggerState::load();
        state.auto_enabled = enabled;
        state.save()
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
```

- [ ] **Step 2: Verify compilation**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p feedback`

---

### Task 8: Integrate SessionLogger into ConversationRuntime

**Files:**
- Modify: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/runtime/Cargo.toml`
- Modify: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/runtime/src/conversation.rs`

- [ ] **Step 1: Add feedback dependency to runtime crate**

In `crates/runtime/Cargo.toml`, add to `[dependencies]`:

```toml
feedback = { path = "../feedback" }
```

- [ ] **Step 2: Add `SessionLogHook` trait to `conversation.rs`**

At the top of `crates/runtime/src/conversation.rs`, after the existing imports, add:

```rust
use feedback::logger::{SessionLogEntry, SessionLogger};
use feedback::quality::{FeedbackSignal, QualityAnalyzer};
```

- [ ] **Step 3: Add optional `SessionLogger` field to `ConversationRuntime`**

In the `ConversationRuntime` struct definition, add a new field:

```rust
    session_logger: Option<SessionLogger>,
    quality_analyzer: QualityAnalyzer,
```

- [ ] **Step 4: Initialize the fields in constructors**

In `new_with_features`, add to the `Self { ... }` block:

```rust
            session_logger: None,
            quality_analyzer: QualityAnalyzer::new(),
```

- [ ] **Step 5: Add builder method for enabling session logging**

After the existing `with_*` methods:

```rust
    #[must_use]
    pub fn with_session_logger(mut self, logger: SessionLogger) -> Self {
        self.session_logger = Some(logger);
        self
    }
```

- [ ] **Step 6: Log messages after each turn completes**

In the `run_turn` method, just before the final `Ok(TurnSummary { ... })` return, add logging logic:

```rust
        // Log session data for feedback pipeline.
        if let Some(logger) = &self.session_logger {
            let _ = logger.ensure_dir();
            let sid = self.session.session_id.clone();
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            // Log the user input.
            let _ = logger.append_entry(&sid, &SessionLogEntry {
                session_id: sid.clone(),
                timestamp_ms: now_ms,
                role: "user".to_string(),
                content: user_input.clone(),
                tool_name: None,
                tool_input: None,
                is_error: None,
            });

            // Log assistant messages and tool results.
            for msg in &assistant_messages {
                for block in &msg.blocks {
                    match block {
                        runtime::session::ContentBlock::Text { text } => {
                            let _ = logger.append_entry(&sid, &SessionLogEntry {
                                session_id: sid.clone(),
                                timestamp_ms: now_ms,
                                role: "assistant".to_string(),
                                content: text.clone(),
                                tool_name: None,
                                tool_input: None,
                                is_error: None,
                            });
                        }
                        runtime::session::ContentBlock::ToolUse { name, input, .. } => {
                            let _ = logger.append_entry(&sid, &SessionLogEntry {
                                session_id: sid.clone(),
                                timestamp_ms: now_ms,
                                role: "assistant".to_string(),
                                content: String::new(),
                                tool_name: Some(name.clone()),
                                tool_input: Some(input.clone()),
                                is_error: None,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
```

- [ ] **Step 7: Verify the workspace compiles**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check --workspace`

---

### Task 9: Add CLI commands for fine-tuning

**Files:**
- Modify: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/commands/src/lib.rs`

- [ ] **Step 1: Add finetune slash command specs**

In the `SLASH_COMMAND_SPECS` array in `crates/commands/src/lib.rs`, add:

```rust
    SlashCommandSpec {
        name: "finetune",
        aliases: &["ft"],
        summary: "Manage fine-tuning pipeline (status/run/rollback/data)",
        argument_hint: Some("[status|run|data stats|data export|rollback|auto --enable|--disable]"),
        resume_supported: false,
    },
```

- [ ] **Step 2: Add finetune command dispatch skeleton**

Add a new public function to `crates/commands/src/lib.rs` that the CLI binary can call:

```rust
/// Dispatch a `claw finetune` subcommand.
pub fn dispatch_finetune(args: &[&str]) -> Result<String, String> {
    match args.first().copied() {
        Some("status") => {
            let orchestrator = feedback::trigger::PipelineOrchestrator::new(
                feedback::FeedbackConfig::default(),
            );
            let status = orchestrator
                .status()
                .map_err(|e| format!("Failed to get status: {e}"))?;
            Ok(format!(
                "Auto-finetune: {}\n\
                 Sessions since last run: {}\n\
                 Last finetune: {}\n\
                 Total session logs: {}\n\
                 Exportable sessions: {}\n\
                 Installed models: {}",
                if status.auto_enabled { "enabled" } else { "disabled" },
                status.sessions_since_last,
                status.last_finetune_timestamp
                    .map_or("never".to_string(), |ts| format!("{ts}")),
                status.total_session_logs,
                status.exportable_sessions,
                if status.installed_model_versions.is_empty() {
                    "none".to_string()
                } else {
                    status.installed_model_versions.join(", ")
                },
            ))
        }
        Some("data") => match args.get(1).copied() {
            Some("stats") => {
                let orchestrator = feedback::trigger::PipelineOrchestrator::new(
                    feedback::FeedbackConfig::default(),
                );
                let status = orchestrator
                    .status()
                    .map_err(|e| format!("Failed to get stats: {e}"))?;
                Ok(format!(
                    "Total sessions: {}\nExportable: {}",
                    status.total_session_logs, status.exportable_sessions,
                ))
            }
            Some("export") => {
                let orchestrator = feedback::trigger::PipelineOrchestrator::new(
                    feedback::FeedbackConfig::default(),
                );
                let stats = orchestrator
                    .export_data()
                    .map_err(|e| format!("Export failed: {e}"))?;
                Ok(format!(
                    "Exported {} sessions ({} conversations), skipped {}",
                    stats.sessions_exported,
                    stats.conversations_written,
                    stats.sessions_skipped,
                ))
            }
            _ => Err("Usage: finetune data [stats|export]".to_string()),
        },
        Some("run") => Ok("Fine-tuning run must be invoked via async runtime. Use `claw finetune run` from the CLI binary.".to_string()),
        Some("rollback") => Ok("Rollback must be invoked via async runtime. Use `claw finetune rollback` from the CLI binary.".to_string()),
        Some("auto") => match args.get(1).copied() {
            Some("--enable") => {
                let orchestrator = feedback::trigger::PipelineOrchestrator::new(
                    feedback::FeedbackConfig::default(),
                );
                orchestrator
                    .set_auto(true)
                    .map_err(|e| format!("Failed: {e}"))?;
                Ok("Auto fine-tuning enabled.".to_string())
            }
            Some("--disable") => {
                let orchestrator = feedback::trigger::PipelineOrchestrator::new(
                    feedback::FeedbackConfig::default(),
                );
                orchestrator
                    .set_auto(false)
                    .map_err(|e| format!("Failed: {e}"))?;
                Ok("Auto fine-tuning disabled.".to_string())
            }
            _ => Err("Usage: finetune auto [--enable|--disable]".to_string()),
        },
        _ => Err(
            "Usage: finetune [status|run|rollback|data stats|data export|auto --enable|--disable]"
                .to_string(),
        ),
    }
}
```

- [ ] **Step 3: Add feedback dependency to commands crate**

In `crates/commands/Cargo.toml`, add to `[dependencies]`:

```toml
feedback = { path = "../feedback" }
```

- [ ] **Step 4: Verify the workspace compiles**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check --workspace`

---

### Task 10: Add reqwest dependency to feedback crate (for reloader sanity tests)

**Files:**
- Modify: `C:/Users/teampooolingforest/Desktop/chat/claw-code/rust/crates/feedback/Cargo.toml`

- [ ] **Step 1: Add reqwest to feedback Cargo.toml**

The `ModelReloader::sanity_test` method uses `reqwest` to call the Ollama API. Update `crates/feedback/Cargo.toml` dependencies:

```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }
runtime = { path = "../runtime" }
serde = { version = "1", features = ["derive"] }
serde_json.workspace = true
tokio = { version = "1", features = ["fs", "io-util", "macros", "process", "rt", "time"] }
```

- [ ] **Step 2: Verify compilation**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check -p feedback`

---

### Task 11: Run full workspace verification

**Files:** All modified and created files.

- [ ] **Step 1: Format the workspace**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo fmt --all`

- [ ] **Step 2: Run clippy on the workspace**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo clippy --workspace --all-targets -- -D warnings`

Fix any warnings reported.

- [ ] **Step 3: Run all tests**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test --workspace`

- [ ] **Step 4: Verify the feedback crate tests specifically**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p feedback -- --nocapture`

---

## Summary of File Changes

| # | Action | File Path | Lines (est.) |
|---|--------|-----------|-------------|
| 1 | Create | `rust/crates/feedback/Cargo.toml` | 18 |
| 2 | Create | `rust/crates/feedback/src/lib.rs` | 95 |
| 3 | Create | `rust/crates/feedback/src/quality.rs` | 100 |
| 4 | Create | `rust/crates/feedback/src/logger.rs` | 140 |
| 5 | Create | `rust/crates/feedback/src/exporter.rs` | 130 |
| 6 | Create | `rust/crates/feedback/src/runner.rs` | 220 |
| 7 | Create | `rust/crates/feedback/src/reloader.rs` | 240 |
| 8 | Create | `rust/crates/feedback/src/trigger.rs` | 200 |
| 9 | Create | `rust/crates/feedback/tests/quality_tests.rs` | 65 |
| 10 | Create | `rust/crates/feedback/tests/logger_tests.rs` | 90 |
| 11 | Create | `rust/crates/feedback/tests/exporter_tests.rs` | 120 |
| 12 | Modify | `rust/crates/runtime/Cargo.toml` | +1 |
| 13 | Modify | `rust/crates/runtime/src/conversation.rs` | +45 |
| 14 | Modify | `rust/crates/commands/src/lib.rs` | +85 |
| 15 | Modify | `rust/crates/commands/Cargo.toml` | +1 |

**Total estimated new code:** ~1,550 lines across 15 files.
