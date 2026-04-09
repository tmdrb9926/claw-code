use gateway::filter::EndpointFilter;

#[test]
fn allows_chat_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/api/chat"));
}

#[test]
fn allows_generate_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/api/generate"));
}

#[test]
fn allows_show_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/api/show"));
}

#[test]
fn allows_ps_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("GET", "/api/ps"));
}

#[test]
fn allows_tags_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("GET", "/api/tags"));
}

#[test]
fn allows_openai_compat_endpoint() {
    let filter = EndpointFilter::default();
    assert!(filter.is_allowed("POST", "/v1/chat/completions"));
}

#[test]
fn blocks_delete_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("DELETE", "/api/delete"));
}

#[test]
fn blocks_create_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/create"));
}

#[test]
fn blocks_pull_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/pull"));
}

#[test]
fn blocks_push_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/push"));
}

#[test]
fn blocks_unknown_endpoint() {
    let filter = EndpointFilter::default();
    assert!(!filter.is_allowed("POST", "/api/something-new"));
}
