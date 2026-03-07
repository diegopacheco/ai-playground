mod routes;
mod k8s;
mod agent;

use axum::Router;
use axum::routing::get;
use axum::routing::post;
use std::sync::Arc;

struct AppState {
    k8s_client: kube::Client,
}

#[tokio::main]
async fn main() {
    let client = kube::Client::try_default().await.expect("Failed to create k8s client");
    let state = Arc::new(AppState { k8s_client: client });

    let app = Router::new()
        .route("/logs", get(routes::logs::get_logs))
        .route("/fix", post(routes::fix::fix_deployments))
        .route("/status", get(routes::status::get_status))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("SRE Agent Operator listening on 0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
