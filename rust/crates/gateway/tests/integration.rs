//! Integration tests for the full gateway stack.
//! These tests start the gateway on a random port with TLS disabled
//! and verify the full request flow.
//!
//! NOTE: Tests that proxy to Ollama require a running Ollama instance.
//! Those are gated behind `#[ignore]` and run with `cargo test -- --ignored`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;

use gateway::keys::KeyStore;
use gateway::ratelimit::RateLimiterMap;
use gateway::AppState;

fn make_state_with_key() -> (AppState, String) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("integration-test", 60).unwrap();
    let secret = key.secret.clone();

    let state = AppState {
        key_store: Arc::new(RwLock::new(store)),
        rate_limiters: Arc::new(RwLock::new(RateLimiterMap::new())),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        http_client: hyper_util::client::legacy::Client::builder(
            hyper_util::rt::TokioExecutor::new(),
        )
        .build_http(),
    };

    (state, secret)
}

#[tokio::test]
async fn health_endpoint_needs_no_auth() {
    let (state, _secret) = make_state_with_key();
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn api_chat_without_auth_returns_401() {
    let (state, _secret) = make_state_with_key();
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/api/chat")
        .method("POST")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn blocked_endpoint_returns_403() {
    let (state, secret) = make_state_with_key();
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/api/delete")
        .method("DELETE")
        .header("Authorization", format!("Bearer {secret}"))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
#[ignore = "Requires running Ollama"]
async fn proxy_to_ollama_tags() {
    let (state, secret) = make_state_with_key();
    let app = gateway::build_router(state);

    let req = Request::builder()
        .uri("/api/tags")
        .method("GET")
        .header("Authorization", format!("Bearer {secret}"))
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
