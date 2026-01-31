use axum::{routing::get, routing::post, Router};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use connect_four_backend::{
    AppState,
    persistence::db::Database,
    routes::{agents, game, history},
    sse::broadcaster::Broadcaster,
};

#[tokio::main]
async fn main() {
    let db = Database::new("connect_four.db").await.expect("Failed to initialize database");
    let broadcaster = Arc::new(RwLock::new(Broadcaster::new()));
    let app_state = AppState {
        db: Arc::new(db),
        broadcaster,
    };
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    let app = Router::new()
        .route("/api/agents", get(agents::list_agents))
        .route("/api/game/start", post(game::start_game))
        .route("/api/game/{id}/stream", get(game::game_stream))
        .route("/api/history", get(history::list_matches))
        .route("/api/history/{id}", get(history::get_match))
        .layer(cors)
        .with_state(app_state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Backend running on http://localhost:8080");
    axum::serve(listener, app).await.unwrap();
}
