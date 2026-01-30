mod claude;
mod handlers;
mod models;

use axum::{
    routing::{get, post},
    Router,
};
use handlers::AppState;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState::new());

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/health", get(handlers::health))
        .route(
            "/api/generate",
            post({
                let state = Arc::clone(&state);
                move |body| handlers::generate_training_handler(state, body)
            }),
        )
        .route(
            "/api/ask",
            post({
                let state = Arc::clone(&state);
                move |body| handlers::ask_question_handler(state, body)
            }),
        )
        .route(
            "/api/submit-quiz",
            post({
                let state = Arc::clone(&state);
                move |body| handlers::submit_quiz_handler(state, body)
            }),
        )
        .route("/api/certificate", post(handlers::generate_certificate_handler))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Server running on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
