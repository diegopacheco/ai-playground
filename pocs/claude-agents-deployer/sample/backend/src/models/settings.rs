use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    pub id: i32,
    pub comments_enabled: bool,
    pub background_theme: String,
    pub updated_at: NaiveDateTime,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateSettings {
    pub comments_enabled: Option<bool>,
    pub background_theme: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_settings_deserialize() {
        let json = r#"{"commentsEnabled":false,"backgroundTheme":"forest"}"#;
        let settings: UpdateSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.comments_enabled, Some(false));
        assert_eq!(settings.background_theme, Some("forest".to_string()));
    }

    #[test]
    fn test_app_settings_serialize_camel_case() {
        let settings = AppSettings {
            id: 1,
            comments_enabled: true,
            background_theme: "classic".to_string(),
            updated_at: NaiveDateTime::parse_from_str("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap(),
        };
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("commentsEnabled"));
        assert!(json.contains("backgroundTheme"));
        assert!(!json.contains("comments_enabled"));
    }
}
