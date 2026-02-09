use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct User {
    pub id: Uuid,
    pub name: String,
    pub email: String,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateUser {
    pub name: String,
    pub email: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_user_deserialize() {
        let json = r#"{"name":"Alice","email":"alice@test.com"}"#;
        let user: CreateUser = serde_json::from_str(json).unwrap();
        assert_eq!(user.name, "Alice");
        assert_eq!(user.email, "alice@test.com");
    }

    #[test]
    fn test_user_serialize_camel_case() {
        let user = User {
            id: Uuid::nil(),
            name: "Alice".to_string(),
            email: "alice@test.com".to_string(),
            created_at: NaiveDateTime::parse_from_str("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap(),
        };
        let json = serde_json::to_string(&user).unwrap();
        assert!(json.contains("createdAt"));
        assert!(!json.contains("created_at"));
        assert!(json.contains("Alice"));
    }

    #[test]
    fn test_user_roundtrip() {
        let user = User {
            id: Uuid::new_v4(),
            name: "Bob".to_string(),
            email: "bob@test.com".to_string(),
            created_at: NaiveDateTime::parse_from_str("2025-06-15 12:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap(),
        };
        let json = serde_json::to_string(&user).unwrap();
        let deserialized: User = serde_json::from_str(&json).unwrap();
        assert_eq!(user, deserialized);
    }
}
