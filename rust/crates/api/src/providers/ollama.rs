use std::collections::VecDeque;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::http_client::build_http_client_or_default;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest,
    MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock, StreamEvent,
    ToolDefinition, ToolResultContentBlock, Usage,
};

use super::{preflight_message_request, Provider, ProviderFuture};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_NUM_CTX: u32 = 32_768;
const DEFAULT_KEEP_ALIVE: &str = "-1";
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(128);
const DEFAULT_MAX_RETRIES: u32 = 8;

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
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .or_else(|_| std::env::var("OLLAMA_HOST"))
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());

        let num_ctx = std::env::var("OLLAMA_NUM_CTX")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(DEFAULT_NUM_CTX);

        let keep_alive = std::env::var("OLLAMA_KEEP_ALIVE")
            .unwrap_or_else(|_| DEFAULT_KEEP_ALIVE.to_string());

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

    #[must_use]
    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = num_ctx;
        self
    }

    #[must_use]
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = keep_alive.into();
        self
    }

    #[must_use]
    pub fn with_retry_policy(
        mut self,
        max_retries: u32,
        initial_backoff: Duration,
        max_backoff: Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.initial_backoff = initial_backoff;
        self.max_backoff = max_backoff;
        self
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let request = MessageRequest {
            stream: false,
            ..request.clone()
        };
        preflight_message_request(&request)?;
        let response = self.send_with_retry(&request, false).await?;
        let body = response.text().await.map_err(ApiError::from)?;
        parse_chat_response(&body, "ollama")
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        preflight_message_request(request)?;
        let stream_request = MessageRequest {
            stream: true,
            ..request.clone()
        };
        let response = self.send_with_retry(&stream_request, true).await?;
        Ok(MessageStream {
            response,
            model: request.model.clone(),
            pending: VecDeque::new(),
            done: false,
            state: StreamState::new(request.model.clone()),
            line_buffer: String::new(),
        })
    }

    async fn send_with_retry(
        &self,
        request: &MessageRequest,
        stream: bool,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;

        let last_error = loop {
            attempts += 1;
            let retryable_error = match self.send_raw_request(request, stream).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                Err(error) => return Err(error),
            };

            if attempts > self.max_retries {
                break retryable_error;
            }

            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        };

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error),
        })
    }

    async fn send_raw_request(
        &self,
        request: &MessageRequest,
        stream: bool,
    ) -> Result<reqwest::Response, ApiError> {
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let body = build_chat_request_value(request, self.num_ctx, &self.keep_alive, stream);
        self.http
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ApiError::from)
    }

    fn backoff_for_attempt(&self, attempt: u32) -> Result<Duration, ApiError> {
        let Some(multiplier) = 1_u32.checked_shl(attempt.saturating_sub(1)) else {
            return Err(ApiError::BackoffOverflow {
                attempt,
                base_delay: self.initial_backoff,
            });
        };
        Ok(self
            .initial_backoff
            .checked_mul(multiplier)
            .map_or(self.max_backoff, |delay| delay.min(self.max_backoff)))
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
// OllamaManager — model management helpers
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

    #[must_use]
    pub fn from_env() -> Self {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .or_else(|_| std::env::var("OLLAMA_HOST"))
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Self::new(base_url)
    }

    /// GET /api/ps — list currently running models.
    pub async fn list_running(&self) -> Result<Value, ApiError> {
        let url = format!("{}/api/ps", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        let body = response.text().await.map_err(ApiError::from)?;
        serde_json::from_str(&body).map_err(|error| {
            ApiError::json_deserialize("Ollama", "manager", &body, error)
        })
    }

    /// POST /api/show — show model information.
    pub async fn show_model(&self, model: &str) -> Result<Value, ApiError> {
        let url = format!("{}/api/show", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .post(&url)
            .json(&json!({ "name": model }))
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        let body = response.text().await.map_err(ApiError::from)?;
        serde_json::from_str(&body).map_err(|error| {
            ApiError::json_deserialize("Ollama", model, &body, error)
        })
    }

    /// GET /api/tags — list locally available models.
    pub async fn list_models(&self) -> Result<Value, ApiError> {
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        let body = response.text().await.map_err(ApiError::from)?;
        serde_json::from_str(&body).map_err(|error| {
            ApiError::json_deserialize("Ollama", "manager", &body, error)
        })
    }
}

// ---------------------------------------------------------------------------
// Request translation (public for testing)
// ---------------------------------------------------------------------------

/// Build the JSON body for an Ollama `/api/chat` request.
/// Returns the JSON as a `String` for easy inspection in tests.
#[must_use]
pub fn build_chat_request(request: &MessageRequest) -> String {
    let value = build_chat_request_value(request, DEFAULT_NUM_CTX, DEFAULT_KEEP_ALIVE, false);
    serde_json::to_string(&value).unwrap_or_default()
}

/// Internal: build the request as a `serde_json::Value`.
fn build_chat_request_value(
    request: &MessageRequest,
    num_ctx: u32,
    keep_alive: &str,
    stream: bool,
) -> Value {
    let mut messages = Vec::new();

    // System prompt becomes the first message
    if let Some(system) = request.system.as_ref().filter(|v| !v.is_empty()) {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }

    // Translate each InputMessage
    for message in &request.messages {
        messages.extend(translate_message(message));
    }

    let mut options = json!({
        "num_ctx": num_ctx,
        "num_predict": request.max_tokens,
    });

    if let Some(temperature) = request.temperature {
        options["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        options["top_p"] = json!(top_p);
    }

    let mut payload = json!({
        "model": request.model,
        "messages": messages,
        "stream": stream,
        "options": options,
        "keep_alive": keep_alive,
    });

    // Tools
    if let Some(tools) = &request.tools {
        payload["tools"] =
            Value::Array(tools.iter().map(ollama_tool_definition).collect::<Vec<_>>());
    }

    payload
}

fn translate_message(message: &InputMessage) -> Vec<Value> {
    match message.role.as_str() {
        "assistant" => {
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for block in &message.content {
                match block {
                    InputContentBlock::Text { text: value } => text.push_str(value),
                    InputContentBlock::ToolUse { id: _, name, input } => tool_calls.push(json!({
                        "function": {
                            "name": name,
                            "arguments": input,
                        }
                    })),
                    InputContentBlock::ToolResult { .. } => {}
                }
            }
            if text.is_empty() && tool_calls.is_empty() {
                Vec::new()
            } else {
                let mut msg = json!({ "role": "assistant" });
                if !text.is_empty() {
                    msg["content"] = json!(text);
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(tool_calls);
                }
                vec![msg]
            }
        }
        _ => message
            .content
            .iter()
            .filter_map(|block| match block {
                InputContentBlock::Text { text } => Some(json!({
                    "role": "user",
                    "content": text,
                })),
                InputContentBlock::ToolResult {
                    content, ..
                } => Some(json!({
                    "role": "tool",
                    "content": flatten_tool_result_content(content),
                })),
                InputContentBlock::ToolUse { .. } => None,
            })
            .collect(),
    }
}

fn flatten_tool_result_content(content: &[ToolResultContentBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            ToolResultContentBlock::Text { text } => text.clone(),
            ToolResultContentBlock::Json { value } => value.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn ollama_tool_definition(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    })
}

// ---------------------------------------------------------------------------
// Response translation (public for testing)
// ---------------------------------------------------------------------------

/// Parse a non-streaming Ollama `/api/chat` response body into a
/// Claw `MessageResponse`.
pub fn parse_chat_response(body: &str, request_id: &str) -> Result<MessageResponse, ApiError> {
    let raw: OllamaChatResponse = serde_json::from_str(body).map_err(|error| {
        ApiError::json_deserialize("Ollama", "unknown", body, error)
    })?;

    let mut content = Vec::new();

    // Text content
    if let Some(text) = raw.message.content.filter(|v| !v.is_empty()) {
        content.push(OutputContentBlock::Text { text });
    }

    // Tool calls
    for (i, tc) in raw.message.tool_calls.iter().enumerate() {
        content.push(OutputContentBlock::ToolUse {
            id: format!("ollama_call_{i}"),
            name: tc.function.name.clone(),
            input: tc.function.arguments.clone(),
        });
    }

    let stop_reason = raw.done_reason.as_deref().map(normalize_done_reason);

    Ok(MessageResponse {
        id: request_id.to_string(),
        kind: "message".to_string(),
        role: raw.message.role,
        content,
        model: raw.model,
        stop_reason: stop_reason.map(ToOwned::to_owned),
        stop_sequence: None,
        usage: Usage {
            input_tokens: raw.prompt_eval_count.unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: raw.eval_count.unwrap_or(0),
        },
        request_id: Some(request_id.to_string()),
    })
}

fn normalize_done_reason(value: &str) -> &str {
    match value {
        "stop" => "end_turn",
        "length" => "max_tokens",
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Ollama native response types (deserialization)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    model: String,
    message: OllamaMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Debug, Deserialize)]
struct OllamaToolCall {
    function: OllamaToolFunction,
}

#[derive(Debug, Deserialize)]
struct OllamaToolFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

// ---------------------------------------------------------------------------
// NDJSON Streaming
// ---------------------------------------------------------------------------

/// Streaming chunk from Ollama's NDJSON `/api/chat` stream.
#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    message: Option<OllamaMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Debug)]
pub struct MessageStream {
    response: reqwest::Response,
    model: String,
    pending: VecDeque<StreamEvent>,
    done: bool,
    state: StreamState,
    line_buffer: String,
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        // Ollama's API does not return request IDs.
        None
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            if self.done {
                self.pending.extend(self.state.finish()?);
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }
                return Ok(None);
            }

            match self.response.chunk().await? {
                Some(chunk) => {
                    let text = String::from_utf8_lossy(&chunk);
                    self.line_buffer.push_str(&text);

                    // Process complete lines (NDJSON: one JSON object per line)
                    while let Some(newline_pos) = self.line_buffer.find('\n') {
                        let line: String = self.line_buffer[..newline_pos].to_string();
                        self.line_buffer = self.line_buffer[newline_pos + 1..].to_string();

                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        let parsed: OllamaStreamChunk =
                            serde_json::from_str(trimmed).map_err(|error| {
                                ApiError::json_deserialize("Ollama", &self.model, trimmed, error)
                            })?;

                        self.pending.extend(self.state.ingest_chunk(parsed)?);
                    }
                }
                None => {
                    // Process any remaining buffer content
                    let remaining = std::mem::take(&mut self.line_buffer);
                    let trimmed = remaining.trim();
                    if !trimmed.is_empty() {
                        let parsed: OllamaStreamChunk =
                            serde_json::from_str(trimmed).map_err(|error| {
                                ApiError::json_deserialize("Ollama", &self.model, trimmed, error)
                            })?;
                        self.pending.extend(self.state.ingest_chunk(parsed)?);
                    }
                    self.done = true;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stream state machine
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct StreamState {
    model: String,
    message_started: bool,
    text_started: bool,
    text_finished: bool,
    finished: bool,
    block_index: u32,
    stop_reason: Option<String>,
    usage: Option<Usage>,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            model,
            message_started: false,
            text_started: false,
            text_finished: false,
            finished: false,
            block_index: 0,
            stop_reason: None,
            usage: None,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn ingest_chunk(&mut self, chunk: OllamaStreamChunk) -> Result<Vec<StreamEvent>, ApiError> {
        let mut events = Vec::new();

        // Emit MessageStart on the first chunk
        if !self.message_started {
            self.message_started = true;
            events.push(StreamEvent::MessageStart(MessageStartEvent {
                message: MessageResponse {
                    id: String::new(),
                    kind: "message".to_string(),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    model: chunk
                        .model
                        .clone()
                        .unwrap_or_else(|| self.model.clone()),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage::default(),
                    request_id: None,
                },
            }));
        }

        // Process message content
        if let Some(ref message) = chunk.message {
            // Text content
            if let Some(ref text) = message.content {
                if !text.is_empty() {
                    if !self.text_started {
                        self.text_started = true;
                        events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                            index: 0,
                            content_block: OutputContentBlock::Text {
                                text: String::new(),
                            },
                        }));
                    }
                    events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                        index: 0,
                        delta: ContentBlockDelta::TextDelta {
                            text: text.clone(),
                        },
                    }));
                }
            }

            // Tool calls in a single chunk
            for (i, tc) in message.tool_calls.iter().enumerate() {
                let idx = self.block_index + 1 + i as u32;
                events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    index: idx,
                    content_block: OutputContentBlock::ToolUse {
                        id: format!("ollama_call_{i}"),
                        name: tc.function.name.clone(),
                        input: tc.function.arguments.clone(),
                    },
                }));
                events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                    index: idx,
                }));
            }
            if !message.tool_calls.is_empty() {
                self.block_index += message.tool_calls.len() as u32;
            }
        }

        // Final chunk (done=true): extract usage and stop_reason
        if chunk.done {
            if let Some(done_reason) = &chunk.done_reason {
                self.stop_reason = Some(normalize_done_reason(done_reason).to_string());
            }
            self.usage = Some(Usage {
                input_tokens: chunk.prompt_eval_count.unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: chunk.eval_count.unwrap_or(0),
            });
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, ApiError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;

        let mut events = Vec::new();

        // Close open text block
        if self.text_started && !self.text_finished {
            self.text_finished = true;
            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: 0,
            }));
        }

        if self.message_started {
            events.push(StreamEvent::MessageDelta(MessageDeltaEvent {
                delta: MessageDelta {
                    stop_reason: Some(
                        self.stop_reason
                            .clone()
                            .unwrap_or_else(|| "end_turn".to_string()),
                    ),
                    stop_sequence: None,
                },
                usage: self.usage.clone().unwrap_or_default(),
            }));
            events.push(StreamEvent::MessageStop(MessageStopEvent {}));
        }

        Ok(events)
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_default();
    let retryable = is_retryable_status(status);

    // Ollama error bodies are typically `{"error": "..."}`.
    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("error")?.as_str().map(ToOwned::to_owned));

    Err(ApiError::Api {
        status,
        error_type: None,
        message,
        request_id: None,
        body,
        retryable,
    })
}

const fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429 | 500 | 502 | 503 | 504)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn build_chat_request_includes_system_as_first_message() {
        let request = MessageRequest {
            model: "gemma4:31b".to_string(),
            max_tokens: 1024,
            messages: vec![InputMessage {
                role: "user".to_string(),
                content: vec![InputContentBlock::Text {
                    text: "hello".to_string(),
                }],
            }],
            system: Some("You are helpful.".to_string()),
            ..Default::default()
        };
        let json_str = build_chat_request(&request);
        let parsed: Value = serde_json::from_str(&json_str).unwrap();
        let messages = parsed["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hello");
    }

    #[test]
    fn build_chat_request_sets_num_predict_in_options() {
        let request = MessageRequest {
            model: "gemma4:31b".to_string(),
            max_tokens: 2048,
            messages: vec![],
            ..Default::default()
        };
        let json_str = build_chat_request(&request);
        let parsed: Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["options"]["num_predict"], 2048);
        assert_eq!(parsed["options"]["num_ctx"], DEFAULT_NUM_CTX);
    }

    #[test]
    fn parse_chat_response_extracts_text_and_usage() {
        let body = serde_json::to_string(&json!({
            "model": "gemma4:31b",
            "message": {
                "role": "assistant",
                "content": "Hello world!"
            },
            "done": true,
            "done_reason": "stop",
            "eval_count": 10,
            "prompt_eval_count": 25
        }))
        .unwrap();

        let response = parse_chat_response(&body, "test-req-1").unwrap();
        assert_eq!(response.model, "gemma4:31b");
        assert_eq!(response.content.len(), 1);
        assert!(matches!(&response.content[0], OutputContentBlock::Text { text } if text == "Hello world!"));
        assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(response.usage.output_tokens, 10);
        assert_eq!(response.usage.input_tokens, 25);
    }

    #[test]
    fn parse_chat_response_maps_length_to_max_tokens() {
        let body = serde_json::to_string(&json!({
            "model": "gemma4:31b",
            "message": {
                "role": "assistant",
                "content": "truncated"
            },
            "done": true,
            "done_reason": "length",
            "eval_count": 100,
            "prompt_eval_count": 50
        }))
        .unwrap();

        let response = parse_chat_response(&body, "req-2").unwrap();
        assert_eq!(response.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn parse_tool_call_response() {
        let body = serde_json::to_string(&json!({
            "model": "gemma4:31b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": { "city": "Berlin" }
                    }
                }]
            },
            "done": true,
            "done_reason": "stop",
            "eval_count": 5,
            "prompt_eval_count": 20
        }))
        .unwrap();

        let response = parse_chat_response(&body, "req-3").unwrap();
        assert_eq!(response.content.len(), 1);
        match &response.content[0] {
            OutputContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "ollama_call_0");
                assert_eq!(name, "get_weather");
                assert_eq!(input, &json!({ "city": "Berlin" }));
            }
            other => panic!("expected ToolUse, got: {other:?}"),
        }
    }
}
