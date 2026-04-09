use gateway::keys::KeyStore;
use tempfile::TempDir;

#[test]
fn create_key_generates_prefixed_secret() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("test-laptop", 30).unwrap();

    assert!(key.secret.starts_with("claw-sk-"));
    assert_eq!(key.name, "test-laptop");
    assert_eq!(key.rate_limit, 30);
    assert!(key.enabled);
}

#[test]
fn revoke_key_disables_it() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("ephemeral", 10).unwrap();
    let id = key.id.clone();

    store.revoke(&id).unwrap();
    assert!(!store.find_by_id(&id).unwrap().enabled);
}

#[test]
fn validate_secret_returns_matching_key() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("my-key", 30).unwrap();
    let secret = key.secret.clone();

    let found = store.validate_secret(&secret);
    assert!(found.is_some());
    assert_eq!(found.unwrap().name, "my-key");
}

#[test]
fn validate_revoked_key_returns_none() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("keys.json");
    let mut store = KeyStore::load_from_path(&path).unwrap();
    let key = store.create_key("revoked", 30).unwrap();
    let secret = key.secret.clone();
    store.revoke(&key.id).unwrap();

    let found = store.validate_secret(&secret);
    assert!(found.is_none());
}
