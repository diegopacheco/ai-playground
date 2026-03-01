pub mod db;
pub mod handlers;
pub mod models;

use axum::{Router, routing::{get, post}};
use sqlx::SqlitePool;
use tower_http::cors::{CorsLayer, AllowOrigin, AllowMethods, AllowHeaders};
use axum::http::{Method, HeaderName};

pub fn create_router(pool: SqlitePool) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::exact(
            "http://localhost:5173".parse().unwrap(),
        ))
        .allow_methods(AllowMethods::list([Method::GET, Method::POST]))
        .allow_headers(AllowHeaders::list([
            HeaderName::from_static("content-type"),
        ]));

    Router::new()
        .route("/api/players", post(handlers::create_player))
        .route("/api/players/{id}/stats", get(handlers::get_player_stats))
        .route("/api/games", post(handlers::create_game))
        .route("/api/games/{id}", get(handlers::get_game))
        .route("/api/games/{id}/flip", post(handlers::flip_card))
        .route("/api/leaderboard", get(handlers::get_leaderboard))
        .layer(cors)
        .with_state(pool)
}
