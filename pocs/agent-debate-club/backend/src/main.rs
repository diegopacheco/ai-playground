use axum::{routing::{get, post}, Router};
use tower_http::cors::{Any, CorsLayer};
use debate_club_backend::persistence::db;
use debate_club_backend::routes::{debate, history};
use debate_club_backend::sse::broadcaster::Broadcaster;
use debate_club_backend::AppState;

#[tokio::main]
async fn main() {
    let pool = db::init_db().await;
    let broadcaster = Broadcaster::new();

    let state = AppState {
        pool,
        broadcaster,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/debates", post(debate::create_debate))
        .route("/api/debates", get(history::get_debates))
        .route("/api/debates/{id}", get(history::get_debate))
        .route("/api/debates/{id}/stream", get(debate::debate_stream))
        .route("/api/agents", get(history::get_agents))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
