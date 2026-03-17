use axum::{Router, routing::{post, get}};
use sqlx::postgres::PgPoolOptions;
use std::sync::Arc;
use tower_http::cors::{CorsLayer, Any};

mod app_state;
mod agents;
mod routes;
mod sql_agent;
mod persistence;

use app_state::AppState;
use routes::{query, history, schema};

#[tokio::main]
async fn main() {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://sqlagent:sqlagent123@localhost:5432/salesdb".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .expect("Failed to connect to PostgreSQL");

    persistence::db::init_db(&pool).await;

    let state = Arc::new(AppState {
        pool,
        broadcaster: app_state::Broadcaster::new(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/query", post(query::create_query))
        .route("/api/query/{id}/stream", get(query::stream_query))
        .route("/api/queries", get(history::get_queries))
        .route("/api/queries/{id}", get(history::get_query))
        .route("/api/schema", get(schema::get_schema))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Backend running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
