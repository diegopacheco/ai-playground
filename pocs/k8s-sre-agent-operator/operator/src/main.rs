mod routes;
mod k8s;
mod agent;
mod history;

use axum::Router;
use axum::routing::get;
use axum::routing::post;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

pub struct AppState {
    pub k8s_client: kube::Client,
    pub history: history::HistoryLog,
}

#[tokio::main]
async fn main() {
    let client = kube::Client::try_default().await.expect("Failed to create k8s client");
    let hist = history::new_history();
    let state = Arc::new(AppState { k8s_client: client, history: hist });

    let spa = ServeDir::new("/app/static")
        .not_found_service(ServeFile::new("/app/static/index.html"));

    let app = Router::new()
        .route("/api/logs", get(routes::logs::get_logs))
        .route("/api/status", get(routes::api_status::get_api_status))
        .route("/api/diagnostics", get(routes::diagnostics::get_diagnostics))
        .route("/api/fix", post(routes::fix::fix_deployments))
        .route("/api/apply", post(routes::apply::apply_yaml))
        .route("/api/history", get(routes::history::get_history))
        .route("/logs", get(routes::logs::get_logs))
        .route("/status", get(routes::status::get_status))
        .route("/diagnostics", get(routes::diagnostics::get_diagnostics))
        .route("/fix", post(routes::fix::fix_deployments))
        .route("/apply", post(routes::apply::apply_yaml))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .fallback_service(spa);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("SRE Agent Operator listening on 0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
