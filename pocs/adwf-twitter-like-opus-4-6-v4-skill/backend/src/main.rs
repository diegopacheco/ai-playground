mod auth;
mod db;
mod errors;
mod handlers;
mod models;

use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    let pool = db::create_pool().await;

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<axum::http::HeaderValue>().unwrap())
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/users", post(handlers::users::register))
        .route("/api/users/login", post(handlers::users::login))
        .route("/api/users/me", get(handlers::users::get_me))
        .route("/api/users/{id}", get(handlers::users::get_user))
        .route("/api/users/{id}/followers", get(handlers::users::get_followers))
        .route("/api/users/{id}/following", get(handlers::users::get_following))
        .route("/api/users/{id}/follow", post(handlers::follows::follow_user))
        .route("/api/users/{id}/follow", delete(handlers::follows::unfollow_user))
        .route("/api/posts", post(handlers::posts::create_post))
        .route("/api/posts", get(handlers::posts::get_all_posts))
        .route("/api/posts/{id}", get(handlers::posts::get_post))
        .route("/api/posts/{id}", delete(handlers::posts::delete_post))
        .route("/api/posts/{id}/like", post(handlers::likes::like_post))
        .route("/api/posts/{id}/like", delete(handlers::likes::unlike_post))
        .route("/api/posts/{id}/likes", get(handlers::likes::get_like_count))
        .route("/api/feed", get(handlers::feed::get_feed))
        .layer(cors)
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Server running on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
