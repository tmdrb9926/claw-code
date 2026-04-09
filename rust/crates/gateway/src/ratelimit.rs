use std::collections::HashMap;
use std::time::Instant;

use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::auth::AuthenticatedKey;
use crate::AppState;

/// Result of a rate limit check.
#[derive(Debug)]
pub enum RateLimitResult {
    Allowed,
    Limited { retry_after_secs: u64 },
}

/// Token bucket for a single key.
#[derive(Debug, Clone)]
struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(requests_per_minute: u32) -> Self {
        let max = f64::from(requests_per_minute);
        Self {
            tokens: max,
            max_tokens: max,
            refill_rate: max / 60.0,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self) -> RateLimitResult {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            RateLimitResult::Allowed
        } else {
            let wait = (1.0 - self.tokens) / self.refill_rate;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            RateLimitResult::Limited {
                retry_after_secs: wait.ceil() as u64,
            }
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }
}

/// Map of `key_id` -> token bucket.
#[derive(Debug)]
pub struct RateLimiterMap {
    buckets: HashMap<String, TokenBucket>,
}

impl RateLimiterMap {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Check rate limit for a key. Creates a bucket on first access.
    pub fn check_and_consume(&mut self, key_id: &str, requests_per_minute: u32) -> RateLimitResult {
        let bucket = self
            .buckets
            .entry(key_id.to_string())
            .or_insert_with(|| TokenBucket::new(requests_per_minute));
        bucket.try_consume()
    }
}

impl Default for RateLimiterMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Axum middleware: enforces per-key rate limits.
/// Must run AFTER auth middleware (needs `AuthenticatedKey` in extensions).
pub async fn rate_limit_middleware(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Skip rate limiting for health endpoint
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let auth_key = match req.extensions().get::<AuthenticatedKey>() {
        Some(k) => k.clone(),
        None => {
            // Auth middleware should have rejected this already, but be safe.
            return (StatusCode::UNAUTHORIZED, "Not authenticated").into_response();
        }
    };

    let result = {
        let mut limiters = state.rate_limiters.write().await;
        limiters.check_and_consume(&auth_key.key_id, auth_key.rate_limit)
    };

    match result {
        RateLimitResult::Allowed => next.run(req).await,
        RateLimitResult::Limited { retry_after_secs } => {
            tracing::warn!(key_id = %auth_key.key_id, "Rate limited");
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("Retry-After", retry_after_secs.to_string())],
                format!("Rate limit exceeded. Retry after {retry_after_secs}s."),
            )
                .into_response()
        }
    }
}
