pub struct Config {
    pub database_url: String,
    pub jwt_secret: String,
    pub server_port: u16,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://twitter:twitter123@localhost:5432/twitter".to_string()),
            jwt_secret: std::env::var("JWT_SECRET")
                .unwrap_or_else(|_| "super-secret-jwt-key-change-in-production".to_string()),
            server_port: std::env::var("SERVER_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_struct_fields() {
        let config = Config {
            database_url: "pg://localhost/test".to_string(),
            jwt_secret: "secret".to_string(),
            server_port: 9090,
        };
        assert_eq!(config.database_url, "pg://localhost/test");
        assert_eq!(config.jwt_secret, "secret");
        assert_eq!(config.server_port, 9090);
    }

    #[test]
    fn test_config_from_env_returns_config() {
        let config = Config::from_env();
        assert!(!config.database_url.is_empty());
        assert!(!config.jwt_secret.is_empty());
        assert!(config.server_port > 0);
    }

    #[test]
    fn test_config_port_is_reasonable() {
        let config = Config::from_env();
        assert!(config.server_port > 0);
    }
}
