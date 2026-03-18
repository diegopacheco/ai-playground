use axum::{Router, routing::post};
use tower_http::cors::{CorsLayer, Any};
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
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Semantic Similarity Eval server running on http://0.0.0.0:3000");
    println!("POST /api/eval         - evaluate single answer");
    println!("POST /api/eval/batch   - evaluate batch of answers");
    axum::serve(listener, app).await.unwrap();
}
