mod api;
mod config;
mod ingest;
mod ollama;
mod rag;
mod store;
mod youtube;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = config::Config::from_env();
    match std::env::args().nth(1).as_deref() {
        Some("serve") => api::serve(config).await,
        Some("ingest") => ingest::run(config).await,
        _ => anyhow::bail!("usage: videos-rag serve|ingest"),
    }
}
