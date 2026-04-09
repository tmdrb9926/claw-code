use std::fs;
use tempfile::TempDir;

use feedback::exporter::DataExporter;
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
                SessionLogEntry {
                    session_id: "s1".to_string(),
                    timestamp_ms: 1,
                    role: "user".to_string(),
                    content: "a".to_string(),
                    tool_name: None,
                    tool_input: None,
                    is_error: None,
                },
                SessionLogEntry {
                    session_id: "s1".to_string(),
                    timestamp_ms: 2,
                    role: "assistant".to_string(),
                    content: "b".to_string(),
                    tool_name: None,
                    tool_input: None,
                    is_error: None,
                },
            ],
            feedback: Some("accepted".to_string()),
        },
        SessionLog {
            session_id: "s2".to_string(),
            entries: vec![
                SessionLogEntry {
                    session_id: "s2".to_string(),
                    timestamp_ms: 3,
                    role: "user".to_string(),
                    content: "c".to_string(),
                    tool_name: None,
                    tool_input: None,
                    is_error: None,
                },
                SessionLogEntry {
                    session_id: "s2".to_string(),
                    timestamp_ms: 4,
                    role: "assistant".to_string(),
                    content: "d".to_string(),
                    tool_name: None,
                    tool_input: None,
                    is_error: None,
                },
            ],
            feedback: Some("accepted".to_string()),
        },
    ];

    let exporter = DataExporter::new();
    let stats = exporter.export_to_jsonl(&logs, &output).unwrap();
    assert_eq!(stats.sessions_exported, 2);
    assert_eq!(stats.conversations_written, 2);
}
