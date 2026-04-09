/// Allowlist-based endpoint filter. Only explicitly listed routes are permitted.
/// This is a security-critical component -- default-deny.
#[derive(Debug, Clone)]
pub struct EndpointFilter {
    allowed: Vec<AllowedRoute>,
}

#[derive(Debug, Clone)]
struct AllowedRoute {
    method: &'static str,
    path: &'static str,
}

impl Default for EndpointFilter {
    fn default() -> Self {
        Self {
            allowed: vec![
                AllowedRoute {
                    method: "POST",
                    path: "/api/chat",
                },
                AllowedRoute {
                    method: "POST",
                    path: "/api/generate",
                },
                AllowedRoute {
                    method: "POST",
                    path: "/api/show",
                },
                AllowedRoute {
                    method: "GET",
                    path: "/api/ps",
                },
                AllowedRoute {
                    method: "GET",
                    path: "/api/tags",
                },
                AllowedRoute {
                    method: "POST",
                    path: "/v1/chat/completions",
                },
            ],
        }
    }
}

impl EndpointFilter {
    /// Returns `true` if the given method + path combination is on the allowlist.
    #[must_use]
    pub fn is_allowed(&self, method: &str, path: &str) -> bool {
        // Normalize: strip trailing slash
        let path = path.trim_end_matches('/');
        self.allowed
            .iter()
            .any(|r| r.method.eq_ignore_ascii_case(method) && r.path == path)
    }
}
