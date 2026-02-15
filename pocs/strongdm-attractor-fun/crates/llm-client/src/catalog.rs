#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: &'static str,
    pub provider: &'static str,
    pub context_window: u32,
    pub max_output: u32,
}

pub static MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "gpt-4o",
        provider: "openai",
        context_window: 128000,
        max_output: 16384,
    },
    ModelInfo {
        id: "gpt-4o-mini",
        provider: "openai",
        context_window: 128000,
        max_output: 16384,
    },
    ModelInfo {
        id: "o3-mini",
        provider: "openai",
        context_window: 200000,
        max_output: 100000,
    },
    ModelInfo {
        id: "claude-sonnet-4-20250514",
        provider: "anthropic",
        context_window: 200000,
        max_output: 8192,
    },
    ModelInfo {
        id: "claude-haiku-3-5-20241022",
        provider: "anthropic",
        context_window: 200000,
        max_output: 8192,
    },
    ModelInfo {
        id: "gemini-2.0-flash",
        provider: "gemini",
        context_window: 1048576,
        max_output: 8192,
    },
    ModelInfo {
        id: "gemini-2.5-pro-preview-05-06",
        provider: "gemini",
        context_window: 1048576,
        max_output: 65536,
    },
];

pub fn lookup(model_id: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.id == model_id)
}

pub fn provider_for_model(model_id: &str) -> Option<&'static str> {
    lookup(model_id).map(|m| m.provider)
}
