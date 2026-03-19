mod models;
mod db;
mod agents;
mod engine;
mod handlers;
mod sse;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use handlers::AppState;
use std::sync::Arc;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let database = Arc::new(db::Database::new("werewolf.db"));
    let broadcaster = Arc::new(sse::Broadcaster::new());

    let state = web::Data::new(AppState {
        db: database,
        broadcaster,
    });

    println!("Werewolf server running on http://localhost:3000");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(cors)
            .app_data(state.clone())
            .route("/api/games", web::post().to(handlers::create_game))
            .route("/api/games", web::get().to(handlers::list_games))
            .route("/api/games/{id}", web::get().to(handlers::get_game))
            .route("/api/games/{id}/stream", web::get().to(handlers::stream_game))
            .route("/api/agents", web::get().to(handlers::list_agents))
    })
    .bind("0.0.0.0:3000")?
    .run()
    .await
}
