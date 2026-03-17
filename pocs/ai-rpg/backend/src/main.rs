mod routes;
mod agents;
mod game;
mod persistence;
mod sse;

use axum::Router;
use sqlx::sqlite::SqlitePoolOptions;
use std::sync::Arc;
use tower_http::cors::{CorsLayer, Any};

pub struct AppState {
    pub db: sqlx::SqlitePool,
    pub broadcaster: Arc<sse::broadcaster::Broadcaster>,
}

#[tokio::main]
async fn main() {
    let db = SqlitePoolOptions::new()
        .max_connections(5)
        .connect("sqlite:rpg.db?mode=rwc")
        .await
        .expect("Failed to connect to database");

    persistence::db::init_db(&db).await;

    let state = Arc::new(AppState {
        db,
        broadcaster: Arc::new(sse::broadcaster::Broadcaster::new()),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .nest("/api", routes::api_routes())
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}
