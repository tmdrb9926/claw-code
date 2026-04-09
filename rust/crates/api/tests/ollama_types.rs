use api::providers::ollama::{build_chat_request, parse_chat_response};
use api::{
    InputContentBlock, InputMessage, MessageRequest, OutputContentBlock, ToolDefinition,
};
use serde_json::{json, Value};

#[test]
fn translate_basic_chat_request() {
    let request = MessageRequest {
        model: "gemma4:31b".to_string(),
        max_tokens: 4096,
        messages: vec![InputMessage {
            role: "user".to_string(),
            content: vec![InputContentBlock::Text {
                text: "What is Rust?".to_string(),
            }],
        }],
        system: Some("You are a programming assistant.".to_string()),
        ..Default::default()
    };

    let json_str = build_chat_request(&request);
    let parsed: Value = serde_json::from_str(&json_str).expect("valid JSON");

    // System prompt is the first message
    let messages = parsed["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 2, "system + user = 2 messages");
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "You are a programming assistant.");

    // User message
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "What is Rust?");

    // max_tokens maps to options.num_predict
    assert_eq!(parsed["options"]["num_predict"], 4096);

    // Model is passed through
    assert_eq!(parsed["model"], "gemma4:31b");
}

#[test]
fn translate_tool_definitions() {
    let request = MessageRequest {
        model: "gemma4:31b".to_string(),
        max_tokens: 1024,
        messages: vec![InputMessage {
            role: "user".to_string(),
            content: vec![InputContentBlock::Text {
                text: "What is the weather?".to_string(),
            }],
        }],
        tools: Some(vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a city".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        }]),
        ..Default::default()
    };

    let json_str = build_chat_request(&request);
    let parsed: Value = serde_json::from_str(&json_str).expect("valid JSON");

    let tools = parsed["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 1);

    let tool = &tools[0];
    assert_eq!(tool["type"], "function");
    assert_eq!(tool["function"]["name"], "get_weather");
    assert_eq!(
        tool["function"]["description"],
        "Get the current weather for a city"
    );
    assert_eq!(tool["function"]["parameters"]["type"], "object");
    assert_eq!(
        tool["function"]["parameters"]["properties"]["city"]["type"],
        "string"
    );
}

#[test]
fn translate_tool_result_message() {
    let request = MessageRequest {
        model: "gemma4:31b".to_string(),
        max_tokens: 1024,
        messages: vec![
            // Assistant message with tool use
            InputMessage {
                role: "assistant".to_string(),
                content: vec![InputContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({"city": "Berlin"}),
                }],
            },
            // Tool result
            InputMessage {
                role: "user".to_string(),
                content: vec![InputContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![api::ToolResultContentBlock::Text {
                        text: "Sunny, 22C".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ],
        ..Default::default()
    };

    let json_str = build_chat_request(&request);
    let parsed: Value = serde_json::from_str(&json_str).expect("valid JSON");
    let messages = parsed["messages"].as_array().expect("messages array");

    // Assistant message with tool_calls
    assert_eq!(messages[0]["role"], "assistant");
    let tool_calls = messages[0]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
    assert_eq!(tool_calls[0]["function"]["arguments"]["city"], "Berlin");

    // Tool result message
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["content"], "Sunny, 22C");
}

#[test]
fn parse_basic_chat_response() {
    let body = serde_json::to_string(&json!({
        "model": "gemma4:31b",
        "message": {
            "role": "assistant",
            "content": "Rust is a systems programming language."
        },
        "done": true,
        "done_reason": "stop",
        "eval_count": 42,
        "prompt_eval_count": 100
    }))
    .unwrap();

    let response = parse_chat_response(&body, "test-request-id").unwrap();

    // Model preserved
    assert_eq!(response.model, "gemma4:31b");

    // Text content extracted
    assert_eq!(response.content.len(), 1);
    match &response.content[0] {
        OutputContentBlock::Text { text } => {
            assert_eq!(text, "Rust is a systems programming language.");
        }
        other => panic!("expected Text block, got: {other:?}"),
    }

    // stop_reason mapped: "stop" -> "end_turn"
    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));

    // Usage extracted from eval_count / prompt_eval_count
    assert_eq!(response.usage.output_tokens, 42);
    assert_eq!(response.usage.input_tokens, 100);
    assert_eq!(response.usage.cache_creation_input_tokens, 0);
    assert_eq!(response.usage.cache_read_input_tokens, 0);

    // Request ID preserved
    assert_eq!(response.request_id.as_deref(), Some("test-request-id"));
}

#[test]
fn parse_tool_call_response() {
    let body = serde_json::to_string(&json!({
        "model": "gemma4:31b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": { "city": "Tokyo" }
                    }
                },
                {
                    "function": {
                        "name": "get_time",
                        "arguments": { "timezone": "JST" }
                    }
                }
            ]
        },
        "done": true,
        "done_reason": "stop",
        "eval_count": 15,
        "prompt_eval_count": 80
    }))
    .unwrap();

    let response = parse_chat_response(&body, "req-tool").unwrap();

    // Empty content string should be skipped; only tool calls remain
    assert_eq!(response.content.len(), 2);

    match &response.content[0] {
        OutputContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "ollama_call_0");
            assert_eq!(name, "get_weather");
            assert_eq!(input, &json!({ "city": "Tokyo" }));
        }
        other => panic!("expected ToolUse, got: {other:?}"),
    }

    match &response.content[1] {
        OutputContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "ollama_call_1");
            assert_eq!(name, "get_time");
            assert_eq!(input, &json!({ "timezone": "JST" }));
        }
        other => panic!("expected ToolUse, got: {other:?}"),
    }
}
