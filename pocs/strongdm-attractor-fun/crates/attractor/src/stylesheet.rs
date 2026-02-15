use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stylesheet {
    pub default: Option<ModelConfig>,
    pub nodes: HashMap<String, ModelConfig>,
}

impl Stylesheet {
    pub fn new() -> Self {
        Self {
            default: None,
            nodes: HashMap::new(),
        }
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("failed to parse stylesheet: {}", e))
    }

    pub fn config_for(&self, node_id: &str) -> ModelConfig {
        let node_cfg = self.nodes.get(node_id);
        let def = self.default.as_ref();
        ModelConfig {
            model: node_cfg
                .and_then(|c| c.model.clone())
                .or_else(|| def.and_then(|d| d.model.clone())),
            temperature: node_cfg
                .and_then(|c| c.temperature)
                .or_else(|| def.and_then(|d| d.temperature)),
            max_tokens: node_cfg
                .and_then(|c| c.max_tokens)
                .or_else(|| def.and_then(|d| d.max_tokens)),
            system_prompt: node_cfg
                .and_then(|c| c.system_prompt.clone())
                .or_else(|| def.and_then(|d| d.system_prompt.clone())),
        }
    }
}
