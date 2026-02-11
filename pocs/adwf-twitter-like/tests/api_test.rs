use sqlx::postgres::PgPoolOptions;
use twitter_clone::{config::Config, models::*, state::AppState, routes::create_routes};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use std::sync::Arc;

async fn setup_test_state() -> Arc<AppState> {
    let config = Config {
        database_url: std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/twitter_test".to_string()),
        jwt_secret: "test-secret".to_string(),
    };

    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&config.database_url)
        .await
        .expect("Failed to connect to test database");

    sqlx::query("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        .execute(&pool)
        .await
        .ok();

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("Failed to run migrations");

    Arc::new(AppState::new(pool, config))
}

#[tokio::test]
async fn test_user_registration_and_login() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let register_body = serde_json::json!({
        "username": "testuser",
        "email": "test@test.com",
        "password": "password123"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/register")
                .header("content-type", "application/json")
                .body(Body::from(register_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let login_body = serde_json::json!({
        "username": "testuser",
        "password": "password123"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/login")
                .header("content-type", "application/json")
                .body(Body::from(login_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
