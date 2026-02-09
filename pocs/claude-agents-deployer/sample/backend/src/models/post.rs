use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow, PartialEq)]
pub struct Post {
    pub id: Uuid,
    pub title: String,
    pub content: String,
    pub author: String,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreatePost {
    pub title: String,
    pub content: String,
    pub author: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpdatePost {
    pub title: Option<String>,
    pub content: Option<String>,
    pub author: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_post_deserialize() {
        let json = r#"{"title":"My Post","content":"Hello","author":"Alice"}"#;
        let post: CreatePost = serde_json::from_str(json).unwrap();
        assert_eq!(post.title, "My Post");
        assert_eq!(post.content, "Hello");
        assert_eq!(post.author, "Alice");
    }

    #[test]
    fn test_update_post_partial() {
        let json = r#"{"title":"New Title"}"#;
        let update: UpdatePost = serde_json::from_str(json).unwrap();
        assert_eq!(update.title, Some("New Title".to_string()));
        assert_eq!(update.content, None);
        assert_eq!(update.author, None);
    }

    #[test]
    fn test_update_post_all_fields() {
        let json = r#"{"title":"T","content":"C","author":"A"}"#;
        let update: UpdatePost = serde_json::from_str(json).unwrap();
        assert_eq!(update.title, Some("T".to_string()));
        assert_eq!(update.content, Some("C".to_string()));
        assert_eq!(update.author, Some("A".to_string()));
    }

    #[test]
    fn test_post_serialize() {
        let ts = NaiveDateTime::parse_from_str("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let post = Post {
            id: Uuid::nil(),
            title: "Title".to_string(),
            content: "Content".to_string(),
            author: "Author".to_string(),
            created_at: ts,
            updated_at: ts,
        };
        let json = serde_json::to_string(&post).unwrap();
        assert!(json.contains("Title"));
        assert!(json.contains("Content"));
    }

    #[test]
    fn test_post_roundtrip() {
        let ts = NaiveDateTime::parse_from_str("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let post = Post {
            id: Uuid::new_v4(),
            title: "Title".to_string(),
            content: "Content".to_string(),
            author: "Author".to_string(),
            created_at: ts,
            updated_at: ts,
        };
        let json = serde_json::to_string(&post).unwrap();
        let deserialized: Post = serde_json::from_str(&json).unwrap();
        assert_eq!(post, deserialized);
    }
}
