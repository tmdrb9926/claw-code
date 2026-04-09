use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware,
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;

// These tests need the full AppState, so they are integration-level.
// We test the auth extraction logic in isolation here.

#[tokio::test]
async fn missing_auth_header_returns_401() {
    let app = test_app();
    let req = Request::builder()
        .uri("/api/chat")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn invalid_bearer_token_returns_401() {
    let app = test_app();
    let req = Request::builder()
        .uri("/api/chat")
        .header("Authorization", "Bearer invalid-token")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn valid_bearer_token_passes_through() {
    let (app, secret) = test_app_with_key();
    let req = Request::builder()
        .uri("/test")
        .header("Authorization", format!("Bearer {secret}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// Helper: builds a minimal app with auth middleware and no real keys
fn test_app() -> Router {
    use gateway::auth::auth_middleware;
    use gateway::keys::KeyStore;
    use gateway::ratelimit::RateLimiterMap;
    use gateway::AppState;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("claw-test-no-keys.json");
    let store = KeyStore::load_from_path(&path).unwrap();

    let state = AppState {
        key_store: Arc::new(RwLock::new(store)),
        rate_limiters: Arc::new(RwLock::new(RateLimiterMap::new())),
        ollama_url: "http://127.0.0.1:11434".to_string(),
        http_client: hyper_util::client::legacy::Client::builder(
            hyper_util::rt::TokioExecutor::new(),
        )
        .build_http(),
    };

    Router::new()
        .route("/api/chat", get(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
}

// Helper: builds an app with one valid key
fn test_app_with_key() -> (Router, String) {
    use gateway::auth::auth_middleware;
    use gateway::keys::KeyStore;
    use gateway::ratelimit::RateLimiterMap;
    use gateway::AppState;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("claw-test-with-key.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("test", 100).unwrap();
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

    let app = Router::new()
        .route("/test", get(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    (app, secret)
}
