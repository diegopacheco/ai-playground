use axum::{routing::{get, post}, Router};
use tower_http::cors::{Any, CorsLayer};
use agent_observability_backend::routes;
use agent_observability_backend::sse::Broadcaster;
use agent_observability_backend::AppState;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let state = AppState {
        traces: Arc::new(Mutex::new(Vec::new())),
        broadcaster: Broadcaster::new(),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/agent/run", post(routes::run_agent))
        .route("/api/traces", get(routes::get_traces))
        .route("/api/traces/{trace_id}", get(routes::get_trace))
        .route("/api/traces/{trace_id}/stream", get(routes::trace_stream))
        .route("/api/agents", get(routes::get_agents))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
