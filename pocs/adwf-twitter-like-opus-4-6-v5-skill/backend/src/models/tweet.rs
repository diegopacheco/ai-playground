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

impl CreateTweetRequest {
    pub fn is_valid(&self) -> bool {
        !self.content.trim().is_empty() && self.content.len() <= 280
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_tweet() {
        let req = CreateTweetRequest { content: "Hello world".to_string() };
        assert!(req.is_valid());
    }

    #[test]
    fn test_empty_tweet_is_invalid() {
        let req = CreateTweetRequest { content: "".to_string() };
        assert!(!req.is_valid());
    }

    #[test]
    fn test_whitespace_only_tweet_is_invalid() {
        let req = CreateTweetRequest { content: "   ".to_string() };
        assert!(!req.is_valid());
    }

    #[test]
    fn test_280_chars_is_valid() {
        let content = "a".repeat(280);
        let req = CreateTweetRequest { content };
        assert!(req.is_valid());
    }

    #[test]
    fn test_281_chars_is_invalid() {
        let content = "a".repeat(281);
        let req = CreateTweetRequest { content };
        assert!(!req.is_valid());
    }

    #[test]
    fn test_single_char_is_valid() {
        let req = CreateTweetRequest { content: "x".to_string() };
        assert!(req.is_valid());
    }
}
