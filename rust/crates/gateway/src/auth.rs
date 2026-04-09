use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::AppState;

/// Extension type injected into the request after successful auth.
/// Downstream handlers can extract this to know which key made the request.
#[derive(Debug, Clone)]
pub struct AuthenticatedKey {
    pub key_id: String,
    pub key_name: String,
    pub rate_limit: u32,
}

/// Axum middleware: extracts `Authorization: Bearer <token>`, validates against
/// the key store, and injects `AuthenticatedKey` into request extensions.
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Skip auth for health endpoint
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => {
            tracing::warn!(
                "Missing or malformed Authorization header from {:?}",
                req.uri()
            );
            return (
                StatusCode::UNAUTHORIZED,
                "Missing or invalid Authorization header. Use: Bearer <api-key>",
            )
                .into_response();
        }
    };

    let store = state.key_store.read().await;
    if let Some(key) = store.validate_secret(token) {
        let auth_key = AuthenticatedKey {
            key_id: key.id.clone(),
            key_name: key.name.clone(),
            rate_limit: key.rate_limit,
        };
        tracing::info!(key_id = %auth_key.key_id, "Authenticated request");
        req.extensions_mut().insert(auth_key);
        drop(store); // release read lock before proceeding
        next.run(req).await
    } else {
        tracing::warn!("Invalid API key attempt");
        (StatusCode::UNAUTHORIZED, "Invalid API key").into_response()
    }
}
