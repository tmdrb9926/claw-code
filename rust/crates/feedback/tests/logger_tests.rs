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
