use agent_service::config::{process_env, read_config, service_port};
use agent_service::workflow::AgentWorkflow;
use restate_sdk::prelude::*;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = read_config(process_env)?;
    let port = service_port(process_env)?;
    let address = SocketAddr::from(([0, 0, 0, 0], port));
    println!("agent-service on port {port} using {} {}", config.provider.name(), config.model);
    HttpServer::new(Endpoint::builder().bind(AgentWorkflow).build())
        .listen_and_serve(address)
        .await;
    Ok(())
}
