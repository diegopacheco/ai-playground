mod agents;
mod band;

use actix_web::{web, App, HttpServer, HttpResponse};
use actix_cors::Cors;
use serde::Deserialize;

#[derive(Deserialize)]
struct ComposeRequest {
    genre: String,
    rounds: Option<usize>,
}

async fn compose(req: web::Json<ComposeRequest>) -> HttpResponse {
    let rounds = req.rounds.unwrap_or(2);
    match band::engine::compose(&req.genre, rounds).await {
        Ok(result) => HttpResponse::Ok().json(result),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "error": e })),
    }
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({ "status": "ok" }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("AI Band server running on http://localhost:8080");
    HttpServer::new(|| {
        let cors = Cors::permissive();
        App::new()
            .wrap(cors)
            .route("/health", web::get().to(health))
            .route("/compose", web::post().to(compose))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
