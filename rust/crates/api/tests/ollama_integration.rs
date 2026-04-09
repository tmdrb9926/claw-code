//! Integration tests for OllamaClient.
//! These require a running Ollama server with a model loaded.
//! Run with: cargo test --test ollama_integration -- --ignored

use api::providers::ollama::{OllamaClient, OllamaManager};
use api::{InputMessage, MessageRequest};

fn make_test_client() -> OllamaClient {
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_string());
    OllamaClient::new(model)
}

async fn ollama_available() -> bool {
    let base_url = std::env::var("OLLAMA_BASE_URL")
        .or_else(|_| std::env::var("OLLAMA_HOST"))
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    reqwest::get(format!("{base_url}/api/tags")).await.is_ok()
}

#[tokio::test]
#[ignore]
async fn send_basic_message() {
    if !ollama_available().await {
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
    if !ollama_available().await {
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
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let base_url = std::env::var("OLLAMA_BASE_URL")
        .or_else(|_| std::env::var("OLLAMA_HOST"))
        .unwrap_or_else(|_| "http://localhost:11434".to_string());

    let manager = OllamaManager::new(base_url);
    let models = manager.list_models().await.unwrap();
    println!("Models: {models}");
    assert!(models["models"].is_array());
}
