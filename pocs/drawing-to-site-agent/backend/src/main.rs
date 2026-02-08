use axum::{routing::{get, post, delete}, Router};
use tower_http::cors::{Any, CorsLayer};
use drawing_to_site_backend::persistence::db;
use drawing_to_site_backend::routes::{projects, engines};
use drawing_to_site_backend::sse::broadcaster::Broadcaster;
use drawing_to_site_backend::AppState;

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
        .route("/api/engines", get(engines::get_engines))
        .route("/api/projects", post(projects::create_project))
        .route("/api/projects", get(projects::get_projects))
        .route("/api/projects/{id}", get(projects::get_project))
        .route("/api/projects/{id}", delete(projects::delete_project))
        .route("/api/projects/{id}/stream", get(projects::project_stream))
        .nest_service("/output", tower_http::services::ServeDir::new("output"))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
