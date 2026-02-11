use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use time::OffsetDateTime;
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct User {
    pub id: i32,
    pub username: String,
    pub email: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub display_name: Option<String>,
    pub bio: Option<String>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Deserialize, Validate)]
pub struct RegisterRequest {
    #[validate(length(min = 3, max = 50))]
    pub username: String,
    #[validate(email)]
    pub email: String,
    #[validate(length(min = 6))]
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub token: String,
    pub user: User,
}

#[derive(Debug, Deserialize, Validate)]
pub struct UpdateUserRequest {
    #[validate(length(max = 100))]
    pub display_name: Option<String>,
    #[validate(length(max = 500))]
    pub bio: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UserProfile {
    #[serde(flatten)]
    pub user: User,
    pub followers_count: i64,
    pub following_count: i64,
    pub tweets_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_request_validates_username_length() {
        let request = RegisterRequest {
            username: "ab".to_string(),
            email: "test@test.com".to_string(),
            password: "password123".to_string(),
        };
        assert!(request.validate().is_err());

        let request = RegisterRequest {
            username: "abc".to_string(),
            email: "test@test.com".to_string(),
            password: "password123".to_string(),
        };
        assert!(request.validate().is_ok());

        let request = RegisterRequest {
            username: "a".repeat(51),
            email: "test@test.com".to_string(),
            password: "password123".to_string(),
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn register_request_validates_email_format() {
        let request = RegisterRequest {
            username: "testuser".to_string(),
            email: "invalid-email".to_string(),
            password: "password123".to_string(),
        };
        assert!(request.validate().is_err());

        let request = RegisterRequest {
            username: "testuser".to_string(),
            email: "valid@email.com".to_string(),
            password: "password123".to_string(),
        };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn register_request_validates_password_length() {
        let request = RegisterRequest {
            username: "testuser".to_string(),
            email: "test@test.com".to_string(),
            password: "short".to_string(),
        };
        assert!(request.validate().is_err());

        let request = RegisterRequest {
            username: "testuser".to_string(),
            email: "test@test.com".to_string(),
            password: "longenough".to_string(),
        };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn update_user_request_validates_display_name_length() {
        let request = UpdateUserRequest {
            display_name: Some("a".repeat(101)),
            bio: None,
        };
        assert!(request.validate().is_err());

        let request = UpdateUserRequest {
            display_name: Some("Valid Name".to_string()),
            bio: None,
        };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn update_user_request_validates_bio_length() {
        let request = UpdateUserRequest {
            display_name: None,
            bio: Some("a".repeat(501)),
        };
        assert!(request.validate().is_err());

        let request = UpdateUserRequest {
            display_name: None,
            bio: Some("Valid bio".to_string()),
        };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn update_user_request_allows_none_values() {
        let request = UpdateUserRequest {
            display_name: None,
            bio: None,
        };
        assert!(request.validate().is_ok());
    }
}
