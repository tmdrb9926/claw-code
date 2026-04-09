use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use rustls::ServerConfig;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

/// Serve with an auto-generated self-signed certificate.
/// Certificate is cached in `~/.claw/gateway/tls/` for reuse.
pub async fn serve_with_self_signed_tls(addr: SocketAddr, app: Router) -> anyhow::Result<()> {
    let tls_dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot find home dir"))?
        .join(".claw")
        .join("gateway")
        .join("tls");
    fs::create_dir_all(&tls_dir)?;

    let cert_path = tls_dir.join("self-signed.crt");
    let key_path = tls_dir.join("self-signed.key");

    if !cert_path.exists() || !key_path.exists() {
        tracing::info!("Generating self-signed TLS certificate...");
        generate_self_signed(&cert_path, &key_path)?;
    }

    serve_with_tls_files(
        addr,
        app,
        cert_path.to_str().unwrap(),
        key_path.to_str().unwrap(),
    )
    .await
}

/// Serve with TLS using provided certificate and key files.
pub async fn serve_with_tls_files(
    addr: SocketAddr,
    app: Router,
    cert_path: &str,
    key_path: &str,
) -> anyhow::Result<()> {
    let cert_pem = fs::read(cert_path)?;
    let key_pem = fs::read(key_path)?;

    let certs = rustls_pemfile::certs(&mut &cert_pem[..]).collect::<Result<Vec<_>, _>>()?;
    let key = rustls_pemfile::private_key(&mut &key_pem[..])?
        .ok_or_else(|| anyhow::anyhow!("No private key found in {key_path}"))?;

    let tls_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;

    let acceptor = TlsAcceptor::from(Arc::new(tls_config));
    let listener = TcpListener::bind(addr).await?;

    tracing::info!("TLS server listening on {addr}");

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        let acceptor = acceptor.clone();
        let app = app.clone();

        tokio::spawn(async move {
            match acceptor.accept(stream).await {
                Ok(tls_stream) => {
                    let io = hyper_util::rt::TokioIo::new(tls_stream);
                    let service = hyper_util::service::TowerToHyperService::new(app);
                    if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                        hyper_util::rt::TokioExecutor::new(),
                    )
                    .serve_connection(io, service)
                    .await
                    {
                        tracing::error!("Connection error from {peer_addr}: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("TLS handshake failed from {peer_addr}: {e}");
                }
            }
        });
    }
}

/// Generate a self-signed certificate using rcgen.
fn generate_self_signed(cert_path: &PathBuf, key_path: &PathBuf) -> anyhow::Result<()> {
    use rcgen::{CertificateParams, KeyPair};

    let params = CertificateParams::new(vec!["localhost".to_string(), "0.0.0.0".to_string()])?;

    let key_pair = KeyPair::generate()?;
    let cert = params.self_signed(&key_pair)?;

    fs::write(cert_path, cert.pem())?;
    fs::write(key_path, key_pair.serialize_pem())?;

    tracing::info!("Self-signed cert written to {}", cert_path.display());
    Ok(())
}
