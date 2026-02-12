pub mod auth;
pub mod db;
pub mod errors;
pub mod models;
pub mod routes;

use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use sqlx::SqlitePool;
use tower_http::cors::{Any, CorsLayer};

pub async fn create_test_pool(db_path: &str) -> SqlitePool {
    let url = format!("sqlite:{}?mode=rwc", db_path);
    let pool = SqlitePool::connect(&url).await.expect("Failed to create test pool");

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS likes (
            user_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (user_id, post_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS follows (
            follower_id TEXT NOT NULL,
            following_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (follower_id, following_id),
            FOREIGN KEY (follower_id) REFERENCES users(id),
            FOREIGN KEY (following_id) REFERENCES users(id)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    pool
}

pub fn build_app(pool: SqlitePool) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
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

    Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .layer(cors)
        .with_state(pool)
}
