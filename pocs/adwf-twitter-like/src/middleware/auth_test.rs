
#[cfg(test)]
mod tests {
    use crate::middleware::{create_jwt, verify_jwt, Claims};

    const TEST_SECRET: &str = "test_secret_key_for_testing";

    #[test]
    fn create_jwt_returns_valid_token() {
        let user_id = 123;
        let result = create_jwt(user_id, TEST_SECRET);
        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(!token.is_empty());
        assert!(token.contains('.'));
    }

    #[test]
    fn verify_jwt_validates_correct_token() {
        let user_id = 456;
        let token = create_jwt(user_id, TEST_SECRET).unwrap();
        let result = verify_jwt(&token, TEST_SECRET);
        assert!(result.is_ok());
        let claims = result.unwrap();
        assert_eq!(claims.sub, user_id);
    }

    #[test]
    fn verify_jwt_rejects_invalid_token() {
        let invalid_token = "invalid.token.here";
        let result = verify_jwt(invalid_token, TEST_SECRET);
        assert!(result.is_err());
    }

    #[test]
    fn verify_jwt_rejects_wrong_secret() {
        let user_id = 789;
        let token = create_jwt(user_id, TEST_SECRET).unwrap();
        let result = verify_jwt(&token, "wrong_secret");
        assert!(result.is_err());
    }

    #[test]
    fn verify_jwt_rejects_empty_token() {
        let result = verify_jwt("", TEST_SECRET);
        assert!(result.is_err());
    }

    #[test]
    fn create_jwt_includes_expiration() {
        use time::OffsetDateTime;

        let user_id = 321;
        let token = create_jwt(user_id, TEST_SECRET).unwrap();
        let claims = verify_jwt(&token, TEST_SECRET).unwrap();

        let now = OffsetDateTime::now_utc().unix_timestamp();
        assert!(claims.exp > now);
    }

    #[test]
    fn create_jwt_different_users_generate_different_tokens() {
        let token1 = create_jwt(1, TEST_SECRET).unwrap();
        let token2 = create_jwt(2, TEST_SECRET).unwrap();
        assert_ne!(token1, token2);
    }

    #[test]
    fn claims_deserializes_correctly() {
        let user_id = 555;
        let token = create_jwt(user_id, TEST_SECRET).unwrap();
        let claims = verify_jwt(&token, TEST_SECRET).unwrap();
        assert_eq!(claims.sub, user_id);
        assert!(claims.exp > 0);
    }
}
