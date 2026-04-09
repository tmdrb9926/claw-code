# Ollama Provider Implementation Plan (Phase 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native Ollama provider to Claw Code so it can use Gemma 4 31B (or any Ollama model) as its LLM backend via the `/api/chat` endpoint.

**Architecture:** Add `ProviderKind::Ollama` and `OllamaClient` to the existing provider abstraction. OllamaClient communicates with Ollama's native `/api/chat` API (not OpenAI compat), translates between Claw's normalized types and Ollama's JSON format, and handles NDJSON streaming. No auth is required for local connections.

**Tech Stack:** Rust, reqwest, serde, tokio (all existing dependencies -- no new crates needed for Phase 1)

**Phases overview:**
- **Phase 1 (this plan):** OllamaClient provider -- local Gemma 4 inference via Claw Code
- **Phase 2 (separate plan):** FeedbackPipeline -- session logging + fine-tuning loop
- **Phase 3 (separate plan):** API Gateway -- external access with auth + rate limiting

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `crates/api/src/providers/ollama.rs` | OllamaClient, OllamaManager, NDJSON types, streaming, Provider trait impl |
| Modify | `crates/api/src/providers/mod.rs` | Add `ProviderKind::Ollama`, model registry entries, `metadata_for_model`, `detect_provider_kind` |
| Modify | `crates/api/src/client.rs` | Add `Ollama(OllamaClient)` variant to `ProviderClient`, update factory + dispatch |
| Modify | `crates/api/src/lib.rs` | Export `OllamaClient` |
| Modify | `crates/api/src/error.rs` | Add Ollama-specific error context to `safe_failure_class` |
| Create | `crates/api/tests/ollama_types.rs` | Unit tests for request/response translation |
| Create | `crates/api/tests/ollama_stream.rs` | Unit tests for NDJSON stream parsing |
| Create | `crates/api/tests/ollama_integration.rs` | Integration tests (require running Ollama) |

---

### Task 1: Add ProviderKind::Ollama and model registry

**Files:**
- Modify: `crates/api/src/providers/mod.rs`

- [ ] **Step 1: Write failing test for Ollama model alias resolution**

Add to the bottom of `crates/api/src/providers/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_gemma4_alias() {
        assert_eq!(resolve_model_alias("gemma4"), "gemma4:31b");
    }

    #[test]
    fn resolve_gemma4_31b_alias() {
        assert_eq!(resolve_model_alias("gemma4-31b"), "gemma4:31b");
    }

    #[test]
    fn resolve_gemma4_26b_alias() {
        assert_eq!(resolve_model_alias("gemma4-26b"), "gemma4:26b-a4b");
    }

    #[test]
    fn metadata_for_gemma4_model() {
        let meta = metadata_for_model("gemma4").unwrap();
        assert_eq!(meta.provider, ProviderKind::Ollama);
        assert_eq!(meta.auth_env, "OLLAMA_API_KEY");
        assert_eq!(meta.default_base_url, "http://localhost:11434");
    }

    #[test]
    fn detect_ollama_provider_for_gemma4() {
        // Set the env var so detection works even without Anthropic key
        std::env::set_var("OLLAMA_HOST", "http://localhost:11434");
        let kind = detect_provider_kind("gemma4:31b");
        assert_eq!(kind, ProviderKind::Ollama);
        std::env::remove_var("OLLAMA_HOST");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p api --lib providers::tests -- --nocapture`
Expected: FAIL -- variants and aliases don't exist yet.

- [ ] **Step 3: Add Ollama variant to ProviderKind**

In `crates/api/src/providers/mod.rs`, change the `ProviderKind` enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    Xai,
    OpenAi,
    Ollama,
}
```

- [ ] **Step 4: Add Ollama entries to MODEL_REGISTRY**

In the `MODEL_REGISTRY` array in `crates/api/src/providers/mod.rs`, add after the last existing entry:

```rust
    ("gemma4", ProviderMetadata {
        provider: ProviderKind::Ollama,
        auth_env: "OLLAMA_API_KEY",
        base_url_env: "OLLAMA_BASE_URL",
        default_base_url: "http://localhost:11434",
    }),
    ("gemma4-31b", ProviderMetadata {
        provider: ProviderKind::Ollama,
        auth_env: "OLLAMA_API_KEY",
        base_url_env: "OLLAMA_BASE_URL",
        default_base_url: "http://localhost:11434",
    }),
    ("gemma4-26b", ProviderMetadata {
        provider: ProviderKind::Ollama,
        auth_env: "OLLAMA_API_KEY",
        base_url_env: "OLLAMA_BASE_URL",
        default_base_url: "http://localhost:11434",
    }),
    ("local", ProviderMetadata {
        provider: ProviderKind::Ollama,
        auth_env: "OLLAMA_API_KEY",
        base_url_env: "OLLAMA_BASE_URL",
        default_base_url: "http://localhost:11434",
    }),
```

- [ ] **Step 5: Add Ollama to resolve_model_alias**

In the `resolve_model_alias` function, add the `ProviderKind::Ollama` match arm inside the `find_map` closure:

```rust
                ProviderKind::Ollama => match *alias {
                    "gemma4" | "gemma4-31b" => "gemma4:31b",
                    "gemma4-26b" => "gemma4:26b-a4b",
                    "local" => trimmed,
                    _ => trimmed,
                },
```

- [ ] **Step 6: Add Ollama to metadata_for_model**

In `metadata_for_model`, add before the final `None`:

```rust
    if canonical.starts_with("gemma4") || canonical == "local" {
        return Some(ProviderMetadata {
            provider: ProviderKind::Ollama,
            auth_env: "OLLAMA_API_KEY",
            base_url_env: "OLLAMA_BASE_URL",
            default_base_url: "http://localhost:11434",
        });
    }
```

- [ ] **Step 7: Add Ollama to detect_provider_kind fallback chain**

In `detect_provider_kind`, add before the final `ProviderKind::Anthropic` fallback:

```rust
    if std::env::var("OLLAMA_HOST").is_ok() || std::env::var("OLLAMA_BASE_URL").is_ok() {
        return ProviderKind::Ollama;
    }
```

- [ ] **Step 8: Add Ollama to model_token_limit**

In `model_token_limit`, add a match arm:

```rust
        _ if canonical.starts_with("gemma4") => Some(ModelTokenLimit {
            max_output_tokens: 8_192,
            context_window_tokens: 32_768,
        }),
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p api --lib providers::tests -- --nocapture`
Expected: All 5 tests PASS.

- [ ] **Step 10: Commit**

```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
git add crates/api/src/providers/mod.rs
git commit -m "feat: add ProviderKind::Ollama and model registry entries for Gemma 4"
```

---

### Task 2: Create OllamaClient with Ollama-native request/response types

**Files:**
- Create: `crates/api/src/providers/ollama.rs`
- Modify: `crates/api/src/providers/mod.rs` (add `pub mod ollama;`)

- [ ] **Step 1: Write failing test for request translation**

Create `crates/api/tests/ollama_types.rs`:

```rust
use api::types::{
    InputContentBlock, InputMessage, MessageRequest, ToolChoice, ToolDefinition,
};
use serde_json::json;

#[test]
fn translate_basic_chat_request() {
    let request = MessageRequest {
        model: "gemma4:31b".to_string(),
        max_tokens: 8192,
        messages: vec![InputMessage::user_text("Hello")],
        system: Some("You are a coding assistant.".to_string()),
        stream: false,
        ..Default::default()
    };

    let ollama_body = api::providers::ollama::build_chat_request(&request);
    let body: serde_json::Value = serde_json::from_str(&ollama_body).unwrap();

    assert_eq!(body["model"], "gemma4:31b");
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "You are a coding assistant.");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "Hello");
    assert_eq!(body["options"]["num_predict"], 8192);
}

#[test]
fn translate_tool_definitions() {
    let request = MessageRequest {
        model: "gemma4:31b".to_string(),
        max_tokens: 8192,
        messages: vec![InputMessage::user_text("Read file.rs")],
        tools: Some(vec![ToolDefinition {
            name: "read_file".to_string(),
            description: Some("Read a file".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        }]),
        ..Default::default()
    };

    let ollama_body = api::providers::ollama::build_chat_request(&request);
    let body: serde_json::Value = serde_json::from_str(&ollama_body).unwrap();

    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    assert_eq!(body["tools"][0]["function"]["description"], "Read a file");
}

#[test]
fn translate_tool_result_message() {
    let request = MessageRequest {
        model: "gemma4:31b".to_string(),
        max_tokens: 8192,
        messages: vec![
            InputMessage::user_text("Read file.rs"),
            InputMessage {
                role: "assistant".to_string(),
                content: vec![InputContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path": "file.rs"}),
                }],
            },
            InputMessage::user_tool_result("call_1", "fn main() {}", false),
        ],
        ..Default::default()
    };

    let ollama_body = api::providers::ollama::build_chat_request(&request);
    let body: serde_json::Value = serde_json::from_str(&ollama_body).unwrap();

    // Assistant message with tool calls
    assert_eq!(body["messages"][0]["role"], "assistant");
    assert!(body["messages"][0]["tool_calls"].is_array());
    assert_eq!(body["messages"][0]["tool_calls"][0]["function"]["name"], "read_file");

    // Tool result message
    assert_eq!(body["messages"][1]["role"], "tool");
    assert_eq!(body["messages"][1]["content"], "fn main() {}");
}

#[test]
fn parse_basic_chat_response() {
    let ollama_response = r#"{
        "model": "gemma4:31b",
        "created_at": "2026-04-09T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help?"
        },
        "done": true,
        "done_reason": "stop",
        "total_duration": 1000000000,
        "eval_count": 15,
        "prompt_eval_count": 10
    }"#;

    let response = api::providers::ollama::parse_chat_response(ollama_response, "req_1").unwrap();
    assert_eq!(response.content.len(), 1);
    assert_eq!(response.model, "gemma4:31b");
    assert_eq!(response.stop_reason, Some("end_turn".to_string()));
    assert_eq!(response.usage.output_tokens, 15);
    assert_eq!(response.usage.input_tokens, 10);
}

#[test]
fn parse_tool_call_response() {
    let ollama_response = r#"{
        "model": "gemma4:31b",
        "created_at": "2026-04-09T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "file.rs"}
                    }
                }
            ]
        },
        "done": true,
        "done_reason": "stop",
        "eval_count": 20,
        "prompt_eval_count": 10
    }"#;

    let response = api::providers::ollama::parse_chat_response(ollama_response, "req_1").unwrap();
    assert_eq!(response.content.len(), 1);
    match &response.content[0] {
        api::types::OutputContentBlock::ToolUse { name, input, .. } => {
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "file.rs");
        }
        _ => panic!("Expected ToolUse block"),
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p api --test ollama_types -- --nocapture`
Expected: FAIL -- `ollama` module doesn't exist yet.

- [ ] **Step 3: Create ollama.rs with Ollama-native types**

Create `crates/api/src/providers/ollama.rs`:

```rust
use std::env;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ApiError;
use crate::http_client::build_http_client_or_default;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest,
    MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock, StreamEvent,
    ToolDefinition, Usage,
};

use super::{Provider, ProviderFuture};

pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_MAX_RETRIES: u32 = 3;
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(16);

// ---------------------------------------------------------------------------
// Ollama API request/response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Debug, Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    kind: String,
    function: OllamaToolFunction,
}

#[derive(Debug, Serialize)]
struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    model: String,
    message: OllamaMessage,
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    prompt_eval_count: u32,
}

// ---------------------------------------------------------------------------
// OllamaClient
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    num_ctx: u32,
    keep_alive: String,
    max_retries: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl OllamaClient {
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        let base_url = env::var("OLLAMA_BASE_URL")
            .or_else(|_| env::var("OLLAMA_HOST"))
            .unwrap_or_else(|_| DEFAULT_OLLAMA_BASE_URL.to_string());
        let num_ctx = env::var("OLLAMA_NUM_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32_768);
        let keep_alive = env::var("OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "-1".to_string());

        Self {
            http: build_http_client_or_default(),
            base_url,
            model: model.into(),
            num_ctx,
            keep_alive,
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub async fn send_message(&self, request: &MessageRequest) -> Result<MessageResponse, ApiError> {
        let body = build_chat_request_inner(request, &self.num_ctx, &self.keep_alive, false);
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(ApiError::Http)?;

        let status = response.status();
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(ToOwned::to_owned);

        let response_body = response.text().await.map_err(ApiError::Http)?;

        if !status.is_success() {
            return Err(ApiError::Api {
                status,
                error_type: Some("ollama_error".to_string()),
                message: Some(response_body.clone()),
                request_id,
                body: response_body,
                retryable: status.is_server_error(),
            });
        }

        parse_chat_response(&response_body, request_id.as_deref().unwrap_or(""))
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        let body = build_chat_request_inner(request, &self.num_ctx, &self.keep_alive, true);
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(ApiError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.map_err(ApiError::Http)?;
            return Err(ApiError::Api {
                status,
                error_type: Some("ollama_error".to_string()),
                message: Some(body_text.clone()),
                request_id: None,
                body: body_text,
                retryable: status.is_server_error(),
            });
        }

        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(ToOwned::to_owned);

        Ok(MessageStream {
            response,
            request_id,
            buffer: String::new(),
            started: false,
            block_index: 0,
            done: false,
            model: request.model.clone(),
        })
    }
}

impl Provider for OllamaClient {
    type Stream = MessageStream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse> {
        Box::pin(async move { self.send_message(request).await })
    }

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream> {
        Box::pin(async move { self.stream_message(request).await })
    }
}

// ---------------------------------------------------------------------------
// OllamaManager -- model management
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OllamaManager {
    http: reqwest::Client,
    base_url: String,
}

impl OllamaManager {
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: build_http_client_or_default(),
            base_url: base_url.into(),
        }
    }

    pub async fn list_running(&self) -> Result<Value, ApiError> {
        let url = format!("{}/api/ps", self.base_url);
        let resp = self.http.get(&url).send().await.map_err(ApiError::Http)?;
        resp.json::<Value>().await.map_err(ApiError::Http)
    }

    pub async fn show_model(&self, model: &str) -> Result<Value, ApiError> {
        let url = format!("{}/api/show", self.base_url);
        let body = serde_json::json!({ "model": model });
        let resp = self.http.post(&url).json(&body).send().await.map_err(ApiError::Http)?;
        resp.json::<Value>().await.map_err(ApiError::Http)
    }

    pub async fn list_models(&self) -> Result<Value, ApiError> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.http.get(&url).send().await.map_err(ApiError::Http)?;
        resp.json::<Value>().await.map_err(ApiError::Http)
    }
}

// ---------------------------------------------------------------------------
// NDJSON MessageStream
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MessageStream {
    response: reqwest::Response,
    request_id: Option<String>,
    buffer: String,
    started: bool,
    block_index: u32,
    done: bool,
    model: String,
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        if self.done {
            return Ok(None);
        }

        loop {
            // Try to extract a complete line from buffer
            if let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer[..newline_pos].to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();

                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let chunk: OllamaChatResponse = serde_json::from_str(line).map_err(|e| {
                    ApiError::json_deserialize("Ollama", &self.model, line, e)
                })?;

                return Ok(Some(self.translate_chunk(chunk)));
            }

            // Read more data from response
            let chunk = self.response.chunk().await.map_err(ApiError::Http)?;
            match chunk {
                Some(bytes) => {
                    self.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                None => {
                    // Stream ended
                    if !self.buffer.trim().is_empty() {
                        let remaining = self.buffer.trim().to_string();
                        self.buffer.clear();
                        let chunk: OllamaChatResponse =
                            serde_json::from_str(&remaining).map_err(|e| {
                                ApiError::json_deserialize("Ollama", &self.model, &remaining, e)
                            })?;
                        return Ok(Some(self.translate_chunk(chunk)));
                    }
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }

    fn translate_chunk(&mut self, chunk: OllamaChatResponse) -> StreamEvent {
        if !self.started {
            self.started = true;
            return StreamEvent::MessageStart(MessageStartEvent {
                message: MessageResponse {
                    id: self.request_id.clone().unwrap_or_default(),
                    kind: "message".to_string(),
                    role: "assistant".to_string(),
                    content: vec![],
                    model: chunk.model.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage::default(),
                    request_id: self.request_id.clone(),
                },
            });
        }

        if chunk.done {
            self.done = true;

            // Emit final usage + message stop
            return StreamEvent::MessageDelta(MessageDeltaEvent {
                delta: MessageDelta {
                    stop_reason: Some(translate_done_reason(chunk.done_reason.as_deref())),
                    stop_sequence: None,
                },
                usage: Usage {
                    input_tokens: chunk.prompt_eval_count,
                    output_tokens: chunk.eval_count,
                    ..Default::default()
                },
            });
        }

        // Handle tool calls in streaming
        if let Some(tool_calls) = &chunk.message.tool_calls {
            if let Some(tc) = tool_calls.first() {
                let index = self.block_index;
                self.block_index += 1;
                return StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    index,
                    content_block: OutputContentBlock::ToolUse {
                        id: format!("ollama_call_{index}"),
                        name: tc.function.name.clone(),
                        input: tc.function.arguments.clone(),
                    },
                });
            }
        }

        // Regular text delta
        let text = &chunk.message.content;
        if text.is_empty() {
            return StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: self.block_index,
                delta: ContentBlockDelta::TextDelta {
                    text: String::new(),
                },
            });
        }

        StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: text.clone(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Translation functions (public for testing)
// ---------------------------------------------------------------------------

/// Build the JSON body string for an Ollama `/api/chat` request.
pub fn build_chat_request(request: &MessageRequest) -> String {
    let num_ctx = 32_768u32;
    let keep_alive = "-1".to_string();
    let body = build_chat_request_inner(request, &num_ctx, &keep_alive, request.stream);
    serde_json::to_string(&body).expect("serialization should not fail")
}

fn build_chat_request_inner(
    request: &MessageRequest,
    num_ctx: &u32,
    keep_alive: &str,
    stream: bool,
) -> OllamaChatRequest {
    let mut messages = Vec::new();

    // System prompt as first message
    if let Some(system) = &request.system {
        messages.push(OllamaMessage {
            role: "system".to_string(),
            content: system.clone(),
            tool_calls: None,
        });
    }

    // Translate Claw messages to Ollama format
    for msg in &request.messages {
        translate_input_message(msg, &mut messages);
    }

    // Translate tools
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| OllamaTool {
                kind: "function".to_string(),
                function: OllamaToolFunction {
                    name: t.name.clone(),
                    description: t.description.clone().unwrap_or_default(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect()
    });

    OllamaChatRequest {
        model: request.model.clone(),
        messages,
        stream,
        tools,
        options: Some(OllamaOptions {
            num_predict: Some(request.max_tokens),
            num_ctx: Some(*num_ctx),
            num_gpu: None,
            temperature: request.temperature,
            top_p: request.top_p,
            repeat_penalty: None,
        }),
        keep_alive: Some(keep_alive.to_string()),
    }
}

fn translate_input_message(msg: &InputMessage, out: &mut Vec<OllamaMessage>) {
    match msg.role.as_str() {
        "user" => {
            // Collect text content and tool results separately
            let mut texts = Vec::new();
            for block in &msg.content {
                match block {
                    InputContentBlock::Text { text } => texts.push(text.clone()),
                    InputContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        let result_text = content
                            .iter()
                            .map(|c| match c {
                                crate::types::ToolResultContentBlock::Text { text } => text.clone(),
                                crate::types::ToolResultContentBlock::Json { value } => {
                                    value.to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        out.push(OllamaMessage {
                            role: "tool".to_string(),
                            content: result_text,
                            tool_calls: None,
                        });
                    }
                    _ => {}
                }
            }
            if !texts.is_empty() {
                out.push(OllamaMessage {
                    role: "user".to_string(),
                    content: texts.join("\n"),
                    tool_calls: None,
                });
            }
        }
        "assistant" => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in &msg.content {
                match block {
                    InputContentBlock::Text { text } => text_parts.push(text.clone()),
                    InputContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: name.clone(),
                                arguments: input.clone(),
                            },
                        });
                    }
                    _ => {}
                }
            }

            out.push(OllamaMessage {
                role: "assistant".to_string(),
                content: text_parts.join("\n"),
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            });
        }
        _ => {
            // Pass through unknown roles
            let content = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    InputContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            out.push(OllamaMessage {
                role: msg.role.clone(),
                content,
                tool_calls: None,
            });
        }
    }
}

/// Parse an Ollama `/api/chat` non-streaming response into Claw's `MessageResponse`.
pub fn parse_chat_response(
    body: &str,
    request_id: &str,
) -> Result<MessageResponse, ApiError> {
    let resp: OllamaChatResponse = serde_json::from_str(body).map_err(|e| {
        ApiError::json_deserialize("Ollama", "unknown", body, e)
    })?;

    let mut content = Vec::new();

    // Text content
    if !resp.message.content.is_empty() {
        content.push(OutputContentBlock::Text {
            text: resp.message.content.clone(),
        });
    }

    // Tool calls
    if let Some(tool_calls) = &resp.message.tool_calls {
        for (i, tc) in tool_calls.iter().enumerate() {
            content.push(OutputContentBlock::ToolUse {
                id: format!("ollama_call_{i}"),
                name: tc.function.name.clone(),
                input: tc.function.arguments.clone(),
            });
        }
    }

    Ok(MessageResponse {
        id: request_id.to_string(),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model,
        stop_reason: Some(translate_done_reason(resp.done_reason.as_deref())),
        stop_sequence: None,
        usage: Usage {
            input_tokens: resp.prompt_eval_count,
            output_tokens: resp.eval_count,
            ..Default::default()
        },
        request_id: Some(request_id.to_string()),
    })
}

fn translate_done_reason(reason: Option<&str>) -> String {
    match reason {
        Some("stop") => "end_turn".to_string(),
        Some("length") => "max_tokens".to_string(),
        Some(other) => other.to_string(),
        None => "end_turn".to_string(),
    }
}

// Utility: check if Ollama env vars are set
pub fn has_ollama_env() -> bool {
    env::var("OLLAMA_HOST").is_ok() || env::var("OLLAMA_BASE_URL").is_ok()
}
```

- [ ] **Step 4: Register the module in mod.rs**

Add to `crates/api/src/providers/mod.rs` at the top with other module declarations:

```rust
pub mod ollama;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p api --test ollama_types -- --nocapture`
Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
git add crates/api/src/providers/ollama.rs crates/api/tests/ollama_types.rs
git commit -m "feat: add OllamaClient with native /api/chat request/response translation"
```

---

### Task 3: Write NDJSON stream parsing tests

**Files:**
- Create: `crates/api/tests/ollama_stream.rs`

- [ ] **Step 1: Write streaming tests**

Create `crates/api/tests/ollama_stream.rs`:

```rust
use api::types::StreamEvent;

#[test]
fn parse_ndjson_text_chunks() {
    // Simulate NDJSON lines that Ollama would send
    let lines = vec![
        r#"{"model":"gemma4:31b","message":{"role":"assistant","content":"Hello"},"done":false}"#,
        r#"{"model":"gemma4:31b","message":{"role":"assistant","content":" world"},"done":false}"#,
        r#"{"model":"gemma4:31b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","eval_count":5,"prompt_eval_count":10}"#,
    ];

    let events: Vec<StreamEvent> = lines
        .iter()
        .map(|line| {
            let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
            // Verify each line is valid JSON
            assert!(chunk["model"].is_string());
            chunk
        })
        .collect::<Vec<_>>()
        .len();

    // Basic structural validation that NDJSON lines parse as valid JSON
    assert_eq!(lines.len(), 3);

    // Verify the final chunk has done=true
    let final_chunk: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
    assert_eq!(final_chunk["done"], true);
    assert_eq!(final_chunk["done_reason"], "stop");
    assert_eq!(final_chunk["eval_count"], 5);
}

#[test]
fn parse_ndjson_tool_call_chunk() {
    let line = r#"{"model":"gemma4:31b","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"src/main.rs"}}}]},"done":true,"done_reason":"stop","eval_count":15,"prompt_eval_count":10}"#;

    let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
    assert!(chunk["message"]["tool_calls"].is_array());
    assert_eq!(
        chunk["message"]["tool_calls"][0]["function"]["name"],
        "read_file"
    );
}
```

- [ ] **Step 2: Run tests**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p api --test ollama_stream -- --nocapture`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
git add crates/api/tests/ollama_stream.rs
git commit -m "test: add NDJSON stream parsing tests for Ollama provider"
```

---

### Task 4: Update ProviderClient enum and factory

**Files:**
- Modify: `crates/api/src/client.rs`

- [ ] **Step 1: Add Ollama variant to ProviderClient enum**

In `crates/api/src/client.rs`, update the enum:

```rust
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ProviderClient {
    Anthropic(AnthropicClient),
    Xai(OpenAiCompatClient),
    OpenAi(OpenAiCompatClient),
    Ollama(OllamaClient),
}
```

Add the import at the top:

```rust
use crate::providers::ollama::OllamaClient;
```

- [ ] **Step 2: Update from_model_with_anthropic_auth**

Add the `Ollama` arm to the match in `from_model_with_anthropic_auth`:

```rust
            ProviderKind::Ollama => {
                let resolved = providers::resolve_model_alias(model);
                Ok(Self::Ollama(OllamaClient::new(resolved)))
            }
```

- [ ] **Step 3: Update provider_kind**

Add to `provider_kind()`:

```rust
            Self::Ollama(_) => ProviderKind::Ollama,
```

- [ ] **Step 4: Update send_message**

Add to `send_message()`:

```rust
            Self::Ollama(client) => client.send_message(request).await,
```

- [ ] **Step 5: Update stream_message**

Add the `Ollama` arm to `stream_message()`:

```rust
            Self::Ollama(client) => client
                .stream_message(request)
                .await
                .map(MessageStream::Ollama),
```

- [ ] **Step 6: Update MessageStream enum**

Add the Ollama variant:

```rust
#[derive(Debug)]
pub enum MessageStream {
    Anthropic(anthropic::MessageStream),
    OpenAiCompat(openai_compat::MessageStream),
    Ollama(ollama::MessageStream),
}
```

Add the import:

```rust
use crate::providers::ollama;
```

Update `request_id()`:

```rust
            Self::Ollama(stream) => stream.request_id(),
```

Update `next_event()`:

```rust
            Self::Ollama(stream) => stream.next_event().await,
```

- [ ] **Step 7: Run workspace compilation**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo check --workspace`
Expected: Compilation succeeds.

- [ ] **Step 8: Commit**

```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
git add crates/api/src/client.rs
git commit -m "feat: wire OllamaClient into ProviderClient enum and factory"
```

---

### Task 5: Update lib.rs exports and error context

**Files:**
- Modify: `crates/api/src/lib.rs`
- Modify: `crates/api/src/error.rs`

- [ ] **Step 1: Add OllamaClient export in lib.rs**

Add to the exports in `crates/api/src/lib.rs`:

```rust
pub use providers::ollama::{OllamaClient, OllamaManager};
```

- [ ] **Step 2: Add Ollama to safe_failure_class in error.rs**

In `error.rs`, the `safe_failure_class` method -- no changes needed as the existing error variants already cover Ollama scenarios (Http, Api, Json, etc.). The error path names like "ollama_error" are passed dynamically. No code change required.

- [ ] **Step 3: Run full test suite**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test --workspace`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
git add crates/api/src/lib.rs
git commit -m "feat: export OllamaClient and OllamaManager from api crate"
```

---

### Task 6: Add integration test (requires running Ollama)

**Files:**
- Create: `crates/api/tests/ollama_integration.rs`

- [ ] **Step 1: Write integration test**

Create `crates/api/tests/ollama_integration.rs`:

```rust
//! Integration tests for OllamaClient.
//! These require a running Ollama server with a model loaded.
//! Skip with: cargo test --test ollama_integration -- --ignored

use api::providers::ollama::OllamaClient;
use api::types::{InputMessage, MessageRequest};

fn make_test_client() -> OllamaClient {
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_string());
    OllamaClient::new(model)
}

fn ollama_available() -> bool {
    let base_url = std::env::var("OLLAMA_BASE_URL")
        .or_else(|_| std::env::var("OLLAMA_HOST"))
        .unwrap_or_else(|_| "http://localhost:11434".to_string());

    reqwest::blocking::get(format!("{base_url}/api/tags")).is_ok()
}

#[tokio::test]
#[ignore] // Run with: cargo test --test ollama_integration -- --ignored
async fn send_basic_message() {
    if !ollama_available() {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let client = make_test_client();
    let request = MessageRequest {
        model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_string()),
        max_tokens: 100,
        messages: vec![InputMessage::user_text("Say hello in one word.")],
        ..Default::default()
    };

    let response = client.send_message(&request).await.unwrap();
    assert!(!response.content.is_empty());
    assert_eq!(response.role, "assistant");
    println!("Response: {:?}", response.content);
}

#[tokio::test]
#[ignore]
async fn stream_basic_message() {
    if !ollama_available() {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let client = make_test_client();
    let request = MessageRequest {
        model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_string()),
        max_tokens: 100,
        messages: vec![InputMessage::user_text("Count from 1 to 5.")],
        stream: true,
        ..Default::default()
    };

    let mut stream = client.stream_message(&request).await.unwrap();
    let mut event_count = 0;

    while let Some(event) = stream.next_event().await.unwrap() {
        event_count += 1;
        println!("Event {event_count}: {event:?}");
    }

    assert!(event_count > 0, "Should have received at least one event");
}

#[tokio::test]
#[ignore]
async fn manager_list_models() {
    if !ollama_available() {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let base_url = std::env::var("OLLAMA_BASE_URL")
        .or_else(|_| std::env::var("OLLAMA_HOST"))
        .unwrap_or_else(|_| "http://localhost:11434".to_string());

    let manager = api::providers::ollama::OllamaManager::new(base_url);
    let models = manager.list_models().await.unwrap();
    println!("Models: {models}");
    assert!(models["models"].is_array());
}
```

- [ ] **Step 2: Run unit tests (not integration)**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test -p api -- --nocapture`
Expected: Unit tests PASS. Integration tests are `#[ignore]`d.

- [ ] **Step 3: Run integration tests (if Ollama is running)**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo test --test ollama_integration -- --ignored --nocapture`
Expected: PASS if Ollama is running with a model loaded.

- [ ] **Step 4: Commit**

```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
git add crates/api/tests/ollama_integration.rs
git commit -m "test: add Ollama integration tests (require running server)"
```

---

### Task 7: Update CLI main.rs for Ollama model routing

**Files:**
- Modify: `crates/rusty-claude-cli/src/main.rs`

- [ ] **Step 1: Verify the CLI dispatches through ProviderClient::from_model**

The CLI already uses `ProviderClient::from_model()` which calls `detect_provider_kind()`. Since we updated `detect_provider_kind` in Task 1 to handle Ollama models, the CLI should automatically route `--model gemma4` to OllamaClient.

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && cargo build --workspace`
Expected: Build succeeds.

- [ ] **Step 2: Manual smoke test**

Run: `cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust && ./target/debug/claw --model gemma4 prompt "say hello"`
Expected: Receives response from Ollama/Gemma 4 (requires Ollama running with gemma4:31b loaded).

- [ ] **Step 3: Commit (if any changes were needed)**

```bash
cd C:/Users/teampooolingforest/Desktop/chat/claw-code/rust
git add -A
git commit -m "feat: Ollama provider Phase 1 complete -- local Gemma 4 via claw CLI"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `cargo fmt` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes (unit tests)
- [ ] `cargo test --test ollama_integration -- --ignored` passes (with Ollama running)
- [ ] `claw --model gemma4 prompt "hello"` returns a response
- [ ] `claw --model gemma4` starts an interactive session
- [ ] Existing providers (Anthropic, XAI, OpenAI) still work unchanged
