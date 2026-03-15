mod agents;
mod band;

use actix_web::{web, App, HttpServer, HttpResponse};
use actix_cors::Cors;
use serde::Deserialize;
use tokio::sync::mpsc;
use crate::band::engine::ComposeEvent;

#[derive(Deserialize)]
struct ComposeRequest {
    genre: String,
    rounds: Option<usize>,
}

async fn compose_sse(query: web::Query<ComposeRequest>) -> HttpResponse {
    let genre = query.genre.clone();
    let rounds = query.rounds.unwrap_or(2);

    let (tx, mut rx) = mpsc::channel::<ComposeEvent>(32);

    tokio::spawn(async move {
        band::engine::compose_stream(genre, rounds, tx).await;
    });

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            let json = serde_json::to_string(&event).unwrap_or_default();
            let evt_type = match &event {
                band::engine::ComposeEvent::Thinking { .. } => "thinking",
                band::engine::ComposeEvent::Done { .. } => "done",
                band::engine::ComposeEvent::Final { .. } => "final",
                band::engine::ComposeEvent::Error { .. } => "error",
            };
            yield Ok::<_, actix_web::Error>(
                actix_web::web::Bytes::from(format!("event: {}\ndata: {}\n\n", evt_type, json))
            );
        }
    };

    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(stream)
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
            .route("/compose", web::get().to(compose_sse))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
