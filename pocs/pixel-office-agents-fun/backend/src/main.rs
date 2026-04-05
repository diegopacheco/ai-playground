use pixel_office_backend::AppState;
use pixel_office_backend::persistence::db;
use pixel_office_backend::sse::broadcaster::Broadcaster;
use pixel_office_backend::routes;
use axum::Router;
use tower_http::cors::{CorsLayer, Any};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let pool = db::init_db().await.expect("Failed to init database");
    let broadcaster = Broadcaster::new();
    let state = Arc::new(AppState { pool, broadcaster });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .merge(routes::agents::router())
        .merge(routes::history::router())
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await.unwrap();
    println!("Backend running on http://localhost:3001");
    axum::serve(listener, app).await.unwrap();
}
