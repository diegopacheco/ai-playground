
#[cfg(test)]
mod tests {
    use crate::models::{RegisterRequest, LoginRequest, UpdateUserRequest};
    use validator::Validate;

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
