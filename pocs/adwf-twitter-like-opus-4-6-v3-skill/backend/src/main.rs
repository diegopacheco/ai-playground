mod auth;
mod db;
mod errors;
mod models;
mod routes;

use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let pool = db::create_pool().await.expect("Failed to create database pool");

    tracing::info!("Database initialized");

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<axum::http::HeaderValue>().unwrap())
        .allow_methods(Any)
        .allow_headers(Any);

    let public_routes = Router::new()
        .route("/api/auth/register", post(routes::auth::register))
        .route("/api/auth/login", post(routes::auth::login));

    let protected_routes = Router::new()
        .route("/api/auth/me", get(routes::auth::me))
        .route("/api/users/{id}", get(routes::users::get_user))
        .route("/api/users/{id}/followers", get(routes::users::get_followers))
        .route("/api/users/{id}/following", get(routes::users::get_following))
        .route(
            "/api/users/{id}/follow",
            post(routes::users::follow_user).delete(routes::users::unfollow_user),
        )
        .route("/api/posts", get(routes::posts::get_timeline).post(routes::posts::create_post))
        .route(
            "/api/posts/{id}",
            get(routes::posts::get_post).delete(routes::posts::delete_post),
        )
        .route("/api/posts/{id}/like", post(routes::posts::like_post).delete(routes::posts::unlike_post))
        .route("/api/users/{id}/posts", get(routes::posts::get_user_posts))
        .route_layer(middleware::from_fn(auth::auth_middleware));

    let app = Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .layer(cors)
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind to port 3000");

    tracing::info!("Server running on http://0.0.0.0:3000");

    axum::serve(listener, app)
        .await
        .expect("Server failed");
}

#[cfg(test)]
mod tests {
    use crate::auth::{create_token, verify_token};
    use crate::errors::AppError;
    use crate::models::{User, UserResponse};
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    #[test]
    fn test_create_token_returns_string() {
        let token = create_token("test-user-id").unwrap();
        assert!(!token.is_empty());
    }

    #[test]
    fn test_verify_token_roundtrip() {
        let user_id = "abc-123-def";
        let token = create_token(user_id).unwrap();
        let claims = verify_token(&token).unwrap();
        assert_eq!(claims.sub, user_id);
    }

    #[test]
    fn test_verify_invalid_token_fails() {
        let result = verify_token("not-a-real-jwt-token");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_token_has_future_expiration() {
        let token = create_token("user-1").unwrap();
        let claims = verify_token(&token).unwrap();
        let now = chrono::Utc::now().timestamp() as usize;
        assert!(claims.exp > now);
    }

    #[tokio::test]
    async fn test_bad_request_error_response() {
        let error = AppError::BadRequest("field required".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_unauthorized_error_response() {
        let error = AppError::Unauthorized("invalid token".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_not_found_error_response() {
        let error = AppError::NotFound("not found".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_conflict_error_response() {
        let error = AppError::Conflict("already exists".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn test_internal_error_response() {
        let error = AppError::Internal("server error".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_user_to_user_response_conversion() {
        let user = User {
            id: "id-1".to_string(),
            username: "testuser".to_string(),
            email: "test@test.com".to_string(),
            password_hash: "hashedpw".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let response: UserResponse = user.into();

        assert_eq!(response.id, "id-1");
        assert_eq!(response.username, "testuser");
        assert_eq!(response.email, "test@test.com");
        assert_eq!(response.created_at, "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_user_response_excludes_password() {
        let user = User {
            id: "id-2".to_string(),
            username: "alice".to_string(),
            email: "alice@mail.com".to_string(),
            password_hash: "supersecret".to_string(),
            created_at: "2024-06-15T12:00:00Z".to_string(),
        };

        let response: UserResponse = user.into();
        let json = serde_json::to_string(&response).unwrap();

        assert!(!json.contains("supersecret"));
        assert!(!json.contains("password"));
    }

    #[test]
    fn test_create_token_different_users_differ() {
        let token1 = create_token("user-a").unwrap();
        let token2 = create_token("user-b").unwrap();
        assert_ne!(token1, token2);
    }

    #[tokio::test]
    async fn test_error_response_contains_json_body() {
        let error = AppError::BadRequest("missing field".to_string());
        let response = error.into_response();
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"], "missing field");
    }
}
