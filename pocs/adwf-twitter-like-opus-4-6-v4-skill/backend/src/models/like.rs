use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Like {
    pub id: i64,
    pub user_id: i64,
    pub post_id: i64,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct LikeCountResponse {
    pub post_id: i64,
    pub count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_like_count_response_serialize() {
        let response = LikeCountResponse {
            post_id: 1,
            count: 10,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"post_id\":1"));
        assert!(json.contains("\"count\":10"));
    }
}
