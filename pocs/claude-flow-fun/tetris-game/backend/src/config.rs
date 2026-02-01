use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GameConfig {
    pub theme: String,
    pub board_width: u32,
    pub board_height: u32,
    pub grow_interval: u64,
    pub drop_speed: u64,
    pub freeze_chance: u32,
    pub level_time_multiplier: f64,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            board_width: 10,
            board_height: 20,
            grow_interval: 30000,
            drop_speed: 1000,
            freeze_chance: 2,
            level_time_multiplier: 0.85,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GameConfig::default();
        assert_eq!(config.theme, "dark");
        assert_eq!(config.board_width, 10);
        assert_eq!(config.board_height, 20);
        assert_eq!(config.grow_interval, 30000);
        assert_eq!(config.drop_speed, 1000);
        assert_eq!(config.freeze_chance, 2);
        assert_eq!(config.level_time_multiplier, 0.85);
    }

    #[test]
    fn test_config_serialization() {
        let config = GameConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: GameConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_config_custom_values() {
        let config = GameConfig {
            theme: "neon".to_string(),
            board_width: 12,
            board_height: 25,
            grow_interval: 45000,
            drop_speed: 800,
            freeze_chance: 5,
            level_time_multiplier: 0.9,
        };
        assert_eq!(config.theme, "neon");
        assert_eq!(config.board_width, 12);
        assert_eq!(config.board_height, 25);
    }
}
