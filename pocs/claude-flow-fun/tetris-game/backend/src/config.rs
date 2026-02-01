use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub background_themes: Vec<String>,
    pub level_time_multiplier: f64,
    pub freeze_duration: u64,
    pub board_grow_interval: u64,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            background_themes: vec![
                "classic".to_string(),
                "neon".to_string(),
                "retro".to_string(),
                "dark".to_string(),
                "light".to_string(),
            ],
            level_time_multiplier: 0.85,
            freeze_duration: 500,
            board_grow_interval: 30000,
        }
    }
}
