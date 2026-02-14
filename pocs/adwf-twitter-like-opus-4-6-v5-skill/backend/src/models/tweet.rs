use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use chrono::NaiveDateTime;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct TweetWithAuthor {
    pub id: i32,
    pub user_id: i32,
    pub content: String,
    pub created_at: Option<NaiveDateTime>,
    pub username: String,
    pub display_name: String,
    pub like_count: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct CreateTweetRequest {
    pub content: String,
}
