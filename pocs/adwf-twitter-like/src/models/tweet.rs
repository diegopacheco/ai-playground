use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use time::OffsetDateTime;
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Tweet {
    pub id: i32,
    pub user_id: i32,
    pub content: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Deserialize, Validate)]
pub struct CreateTweetRequest {
    #[validate(length(min = 1, max = 280))]
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct TweetResponse {
    #[serde(flatten)]
    pub tweet: Tweet,
    pub author_username: String,
    pub author_display_name: Option<String>,
    pub likes_count: i64,
    pub retweets_count: i64,
    pub comments_count: i64,
    pub is_liked: bool,
    pub is_retweeted: bool,
}
