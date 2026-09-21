pub struct Config {
    pub port: u16,
    pub redis: String,
    pub ollama: String,
    pub embed_model: String,
    pub claude_model: String,
    pub threshold: f32,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            port: var("PROXY_PORT", "8787").parse().unwrap_or(8787),
            redis: var("REDIS_ADDR", "127.0.0.1:6380"),
            ollama: var("OLLAMA_ADDR", "127.0.0.1:11434"),
            embed_model: var("EMBED_MODEL", "nomic-embed-text"),
            claude_model: var("CLAUDE_MODEL", "claude-sonnet-5"),
            threshold: var("SIMILARITY_THRESHOLD", "0.9").parse().unwrap_or(0.9),
        }
    }
}

fn var(name: &str, fallback: &str) -> String {
    std::env::var(name).ok().filter(|value| !value.trim().is_empty()).unwrap_or_else(|| fallback.to_string())
}
