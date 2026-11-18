use crate::agents::Provider;
use crate::error::AppError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentConfig {
    pub provider: Provider,
    pub model: String,
}

pub fn read_config(env: impl Fn(&str) -> Option<String>) -> Result<AgentConfig, AppError> {
    let raw = env("AGENT_PROVIDER").unwrap_or_default();
    let provider = Provider::parse(&raw).ok_or_else(|| {
        AppError::new(format!("AGENT_PROVIDER must be claude, codex, agy or ollama, got \"{raw}\""))
    })?;
    let model = env("AGENT_MODEL").unwrap_or_default();
    if model.trim().is_empty() {
        return Err(AppError::new("AGENT_MODEL is required"));
    }
    Ok(AgentConfig { provider, model })
}

pub fn service_port(env: impl Fn(&str) -> Option<String>) -> Result<u16, AppError> {
    env("AGENT_SERVICE_PORT")
        .and_then(|port| port.parse::<u16>().ok())
        .filter(|port| *port > 0)
        .ok_or_else(|| AppError::new("AGENT_SERVICE_PORT is required"))
}

pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
