use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Comment {
    pub id: Uuid,
    pub content: String,
    pub author: String,
    pub post_id: Uuid,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateComment {
    pub content: String,
    pub author: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_comment_deserialize() {
        let json = r#"{"content":"Nice post!","author":"Bob"}"#;
        let comment: CreateComment = serde_json::from_str(json).unwrap();
        assert_eq!(comment.content, "Nice post!");
        assert_eq!(comment.author, "Bob");
    }

    #[test]
    fn test_comment_serialize_camel_case() {
        let comment = Comment {
            id: Uuid::nil(),
            content: "Great!".to_string(),
            author: "Bob".to_string(),
            post_id: Uuid::nil(),
            created_at: NaiveDateTime::parse_from_str("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap(),
        };
        let json = serde_json::to_string(&comment).unwrap();
        assert!(json.contains("postId"));
        assert!(json.contains("createdAt"));
        assert!(!json.contains("post_id"));
    }

    #[test]
    fn test_comment_roundtrip() {
        let comment = Comment {
            id: Uuid::new_v4(),
            content: "Content".to_string(),
            author: "Author".to_string(),
            post_id: Uuid::new_v4(),
            created_at: NaiveDateTime::parse_from_str("2025-06-15 12:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap(),
        };
        let json = serde_json::to_string(&comment).unwrap();
        let deserialized: Comment = serde_json::from_str(&json).unwrap();
        assert_eq!(comment, deserialized);
    }
}
