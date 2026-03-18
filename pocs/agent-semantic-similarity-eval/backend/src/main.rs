use axum::{Router, routing::post};
use tower_http::cors::{CorsLayer, Any};
use tower_http::services::ServeDir;
use semantic_similarity_eval::routes::eval;

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/eval", post(eval::evaluate))
        .route("/api/eval/batch", post(eval::batch_evaluate))
        .fallback_service(ServeDir::new("../frontend"))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Semantic Similarity Eval server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
