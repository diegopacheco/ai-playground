use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    pub id: i64,
    pub username: String,
    pub email: String,
    pub password_hash: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserResponse {
    pub id: i64,
    pub username: String,
    pub email: String,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub email: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub user: UserResponse,
}

impl From<User> for UserResponse {
    fn from(user: User) -> Self {
        UserResponse {
            id: user.id,
            username: user.username,
            email: user.email,
            created_at: user.created_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_to_user_response() {
        let user = User {
            id: 1,
            username: "alice".to_string(),
            email: "alice@test.com".to_string(),
            password_hash: "hashed".to_string(),
            created_at: "2024-01-01".to_string(),
        };
        let response = UserResponse::from(user);
        assert_eq!(response.id, 1);
        assert_eq!(response.username, "alice");
        assert_eq!(response.email, "alice@test.com");
        assert_eq!(response.created_at, "2024-01-01");
    }

    #[test]
    fn test_user_response_excludes_password() {
        let user = User {
            id: 2,
            username: "bob".to_string(),
            email: "bob@test.com".to_string(),
            password_hash: "secret_hash".to_string(),
            created_at: "2024-01-02".to_string(),
        };
        let response = UserResponse::from(user);
        let json = serde_json::to_string(&response).unwrap();
        assert!(!json.contains("secret_hash"));
        assert!(!json.contains("password"));
    }

    #[test]
    fn test_create_user_request_deserialize() {
        let json = r#"{"username":"test","email":"test@test.com","password":"pass123"}"#;
        let req: CreateUserRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.username, "test");
        assert_eq!(req.email, "test@test.com");
        assert_eq!(req.password, "pass123");
    }

    #[test]
    fn test_login_request_deserialize() {
        let json = r#"{"username":"alice","password":"secret"}"#;
        let req: LoginRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.username, "alice");
        assert_eq!(req.password, "secret");
    }

    #[test]
    fn test_login_response_serialize() {
        let response = LoginResponse {
            token: "abc123".to_string(),
            user: UserResponse {
                id: 1,
                username: "alice".to_string(),
                email: "alice@test.com".to_string(),
                created_at: "2024-01-01".to_string(),
            },
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("abc123"));
        assert!(json.contains("alice"));
    }
}
