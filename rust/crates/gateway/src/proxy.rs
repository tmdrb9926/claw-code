use axum::{
    body::Body,
    extract::{Request, State},
    http::{uri::Uri, StatusCode},
    response::{IntoResponse, Response},
};

use crate::filter::EndpointFilter;
use crate::AppState;

/// Main proxy handler for `/api/*path`.
/// Checks the endpoint filter, then forwards to Ollama.
pub async fn proxy_handler(State(state): State<AppState>, req: Request<Body>) -> Response {
    let method = req.method().as_str().to_uppercase();
    let path = req.uri().path().to_string();

    // Endpoint filtering (default-deny)
    let filter = EndpointFilter::default();
    if !filter.is_allowed(&method, &path) {
        tracing::warn!(method = %method, path = %path, "Blocked endpoint");
        return (
            StatusCode::FORBIDDEN,
            format!("Endpoint not allowed: {method} {path}"),
        )
            .into_response();
    }

    forward_to_ollama(state, req).await
}

/// OpenAI-compatible `/v1/chat/completions` handler.
/// Ollama natively supports this endpoint, so we just proxy it through.
pub async fn openai_compat_handler(State(state): State<AppState>, req: Request<Body>) -> Response {
    forward_to_ollama(state, req).await
}

/// Forward a request to the Ollama backend.
async fn forward_to_ollama(state: AppState, req: Request<Body>) -> Response {
    let ollama_url = &state.ollama_url;
    let path_and_query = req
        .uri()
        .path_and_query()
        .map_or("/", axum::http::uri::PathAndQuery::as_str);

    let target_uri = format!("{ollama_url}{path_and_query}");
    let uri: Uri = match target_uri.parse() {
        Ok(u) => u,
        Err(e) => {
            tracing::error!("Failed to parse target URI: {e}");
            return (StatusCode::BAD_GATEWAY, "Invalid upstream URI").into_response();
        }
    };

    // Build the proxied request
    let (mut parts, body) = req.into_parts();
    parts.uri = uri;

    // Remove hop-by-hop headers
    parts.headers.remove("host");
    parts.headers.remove("authorization"); // Don't forward our auth to Ollama

    let proxied_req = Request::from_parts(parts, body);

    // Forward to Ollama
    match state.http_client.request(proxied_req).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            let body = Body::new(body);
            Response::from_parts(parts, body)
        }
        Err(e) => {
            tracing::error!("Proxy error: {e}");
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to connect to Ollama: {e}"),
            )
                .into_response()
        }
    }
}
