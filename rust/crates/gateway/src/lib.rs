pub mod auth;
pub mod filter;
pub mod keys;
pub mod proxy;
pub mod ratelimit;
pub mod tls;

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::keys::KeyStore;
use crate::ratelimit::RateLimiterMap;

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub key_store: Arc<RwLock<KeyStore>>,
    pub rate_limiters: Arc<RwLock<RateLimiterMap>>,
    pub ollama_url: String,
    pub http_client: hyper_util::client::legacy::Client<
        hyper_util::client::legacy::connect::HttpConnector,
        axum::body::Body,
    >,
}

/// Build the axum Router with all middleware layers.
pub fn build_router(state: AppState) -> axum::Router {
    use axum::routing::{any, get};

    axum::Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/{*path}", any(proxy::proxy_handler))
        .route(
            "/v1/chat/completions",
            axum::routing::post(proxy::openai_compat_handler),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            ratelimit::rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .with_state(state)
}
