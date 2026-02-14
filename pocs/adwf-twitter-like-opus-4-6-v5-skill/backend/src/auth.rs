use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use serde::{Deserialize, Serialize};
use chrono::{Utc, Duration};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: i32,
    pub exp: usize,
}

pub fn create_token(user_id: i32, secret: &str) -> Result<String, jsonwebtoken::errors::Error> {
    let expiration = Utc::now()
        .checked_add_signed(Duration::hours(24))
        .expect("valid timestamp")
        .timestamp() as usize;

    let claims = Claims {
        sub: user_id,
        exp: expiration,
    };

    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_bytes()))
}

pub fn validate_token(token: &str, secret: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )?;
    Ok(token_data.claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_token_returns_ok() {
        let result = create_token(1, "test-secret");
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_token_is_non_empty() {
        let token = create_token(42, "my-secret").unwrap();
        assert!(!token.is_empty());
    }

    #[test]
    fn test_validate_token_success() {
        let secret = "test-secret-key";
        let token = create_token(7, secret).unwrap();
        let claims = validate_token(&token, secret).unwrap();
        assert_eq!(claims.sub, 7);
    }

    #[test]
    fn test_validate_token_wrong_secret_fails() {
        let token = create_token(1, "correct-secret").unwrap();
        let result = validate_token(&token, "wrong-secret");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_token_invalid_token_fails() {
        let result = validate_token("not-a-valid-token", "secret");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_token_different_users_produce_different_tokens() {
        let secret = "shared-secret";
        let token_a = create_token(1, secret).unwrap();
        let token_b = create_token(2, secret).unwrap();
        assert_ne!(token_a, token_b);
    }

    #[test]
    fn test_claims_expiration_is_in_the_future() {
        let secret = "test-secret";
        let token = create_token(1, secret).unwrap();
        let claims = validate_token(&token, secret).unwrap();
        let now = Utc::now().timestamp() as usize;
        assert!(claims.exp > now);
    }
}
