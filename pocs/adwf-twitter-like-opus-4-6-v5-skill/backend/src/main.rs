mod auth;
mod config;
mod db;
mod handlers;
mod middleware;
mod models;

use axum::{
    middleware::from_fn_with_state,
    routing::{delete, get, post, put},
    Router,
};
use sqlx::PgPool;
use tower_http::cors::{Any, CorsLayer};

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub jwt_secret: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = config::Config::from_env();
    let pool = db::create_pool(&config.database_url).await;
    db::run_migrations(&pool).await;

    let state = AppState {
        pool,
        jwt_secret: config.jwt_secret,
    };

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<axum::http::HeaderValue>().unwrap())
        .allow_methods(Any)
        .allow_headers(Any);

    let protected_routes = Router::new()
        .route("/api/tweets", post(handlers::tweet_handler::create_tweet))
        .route("/api/tweets/feed", get(handlers::tweet_handler::get_feed))
        .route("/api/tweets/{id}", delete(handlers::tweet_handler::delete_tweet))
        .route("/api/tweets/{id}/like", post(handlers::like_handler::like_tweet))
        .route("/api/tweets/{id}/like", delete(handlers::like_handler::unlike_tweet))
        .route("/api/users/{id}/follow", post(handlers::follow_handler::follow_user))
        .route("/api/users/{id}/follow", delete(handlers::follow_handler::unfollow_user))
        .route("/api/users/{id}", put(handlers::user_handler::update_user))
        .layer(from_fn_with_state(state.clone(), middleware::auth_middleware::auth_middleware));

    let public_routes = Router::new()
        .route("/api/auth/register", post(handlers::auth_handler::register))
        .route("/api/auth/login", post(handlers::auth_handler::login))
        .route("/api/users/{id}", get(handlers::user_handler::get_user))
        .route("/api/users/{id}/followers", get(handlers::user_handler::get_followers))
        .route("/api/users/{id}/following", get(handlers::user_handler::get_following))
        .route("/api/tweets/{id}", get(handlers::tweet_handler::get_tweet))
        .route("/api/users/{id}/tweets", get(handlers::tweet_handler::get_user_tweets));

    let app = Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .layer(cors)
        .with_state(state);

    let addr = format!("0.0.0.0:{}", config.server_port);
    tracing::info!("Server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
