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
