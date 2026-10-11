use std::env;

pub struct Config {
    pub api_port: u16,
    pub qdrant_grpc_url: String,
    pub ollama_url: String,
    pub embed_model: String,
    pub chat_model: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            api_port: env::var("API_PORT")
                .ok()
                .and_then(|port| port.parse().ok())
                .unwrap_or(8102),
            qdrant_grpc_url: var("QDRANT_GRPC_URL", "http://localhost:6344"),
            ollama_url: var("OLLAMA_URL", "http://localhost:11434"),
            embed_model: var("EMBED_MODEL", "nomic-embed-text"),
            chat_model: var("CHAT_MODEL", "llama3.2"),
        }
    }
}

fn var(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}
