use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

use crate::errors::AppError;

const JWT_SECRET: &str = "twitter-like-super-secret-key-2024";

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: i64,
    pub username: String,
    pub exp: usize,
}

#[allow(dead_code)]
pub struct AuthUser {
    pub user_id: i64,
    pub username: String,
}

pub fn create_token(user_id: i64, username: &str) -> Result<String, AppError> {
    let expiration = chrono::Utc::now()
        .checked_add_signed(chrono::Duration::hours(24))
        .unwrap()
        .timestamp() as usize;

    let claims = Claims {
        sub: user_id,
        username: username.to_string(),
        exp: expiration,
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(JWT_SECRET.as_bytes()),
    )?;

    Ok(token)
}

pub fn validate_token(token: &str) -> Result<Claims, AppError> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(JWT_SECRET.as_bytes()),
        &Validation::default(),
    )
    .map_err(|_| AppError::Unauthorized("Invalid token".to_string()))?;
    Ok(token_data.claims)
}

impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let auth_header = parts
            .headers
            .get("Authorization")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| AppError::Unauthorized("Missing authorization header".to_string()))?;

        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or_else(|| AppError::Unauthorized("Invalid authorization format".to_string()))?;

        let claims = validate_token(token)?;

        Ok(AuthUser {
            user_id: claims.sub,
            username: claims.username,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_token_success() {
        let result = create_token(1, "testuser");
        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(!token.is_empty());
    }

    #[test]
    fn test_create_token_contains_valid_jwt_parts() {
        let token = create_token(42, "alice").unwrap();
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_validate_token_success() {
        let token = create_token(10, "bob").unwrap();
        let claims = validate_token(&token).unwrap();
        assert_eq!(claims.sub, 10);
        assert_eq!(claims.username, "bob");
    }

    #[test]
    fn test_validate_token_invalid() {
        let result = validate_token("invalid.token.here");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_token_tampered() {
        let token = create_token(1, "user").unwrap();
        let tampered = format!("{}x", token);
        let result = validate_token(&tampered);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_token_different_users_produce_different_tokens() {
        let token1 = create_token(1, "alice").unwrap();
        let token2 = create_token(2, "bob").unwrap();
        assert_ne!(token1, token2);
    }

    #[test]
    fn test_token_expiration_is_in_future() {
        let token = create_token(1, "user").unwrap();
        let claims = validate_token(&token).unwrap();
        let now = chrono::Utc::now().timestamp() as usize;
        assert!(claims.exp > now);
    }
}
