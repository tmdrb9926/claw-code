use gateway::ratelimit::{RateLimitResult, RateLimiterMap};

#[test]
fn allows_requests_within_limit() {
    let mut map = RateLimiterMap::new();
    for _ in 0..5 {
        let result = map.check_and_consume("key_01", 10); // 10 req/min
        assert!(matches!(result, RateLimitResult::Allowed));
    }
}

#[test]
fn rejects_requests_over_limit() {
    let mut map = RateLimiterMap::new();
    // Consume all 3 tokens
    for _ in 0..3 {
        map.check_and_consume("key_01", 3);
    }
    let result = map.check_and_consume("key_01", 3);
    assert!(matches!(result, RateLimitResult::Limited { .. }));
}

#[test]
fn different_keys_have_independent_limits() {
    let mut map = RateLimiterMap::new();
    // Exhaust key_01
    for _ in 0..2 {
        map.check_and_consume("key_01", 2);
    }
    let r1 = map.check_and_consume("key_01", 2);
    assert!(matches!(r1, RateLimitResult::Limited { .. }));

    // key_02 should still work
    let r2 = map.check_and_consume("key_02", 2);
    assert!(matches!(r2, RateLimitResult::Allowed));
}
