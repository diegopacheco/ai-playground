mod models;
mod handlers;
mod score_engine;
mod agents;

use axum::{routing::{get, post}, Router};
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/analyze", post(handlers::analyze_stream))
        .route("/api/health", get(handlers::health))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Backend running on http://localhost:8080");
    axum::serve(listener, app).await.unwrap();
}
