use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Post {
    pub id: i64,
    pub user_id: i64,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct PostResponse {
    pub id: i64,
    pub user_id: i64,
    pub content: String,
    pub created_at: String,
    pub username: String,
    pub like_count: i64,
}

#[derive(Debug, Deserialize)]
pub struct CreatePostRequest {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    pub page: Option<i64>,
    pub limit: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post_response_serialize() {
        let response = PostResponse {
            id: 1,
            user_id: 2,
            content: "Hello world".to_string(),
            created_at: "2024-01-01".to_string(),
            username: "alice".to_string(),
            like_count: 5,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("Hello world"));
        assert!(json.contains("alice"));
        assert!(json.contains("5"));
    }

    #[test]
    fn test_create_post_request_deserialize() {
        let json = r#"{"content":"My first post"}"#;
        let req: CreatePostRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.content, "My first post");
    }

    #[test]
    fn test_pagination_params_defaults() {
        let json = r#"{}"#;
        let params: PaginationParams = serde_json::from_str(json).unwrap();
        assert!(params.page.is_none());
        assert!(params.limit.is_none());
    }

    #[test]
    fn test_pagination_params_with_values() {
        let json = r#"{"page":2,"limit":50}"#;
        let params: PaginationParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.page, Some(2));
        assert_eq!(params.limit, Some(50));
    }
}
