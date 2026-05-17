pub mod game;
pub mod routes;
pub mod scores;

use actix_cors::Cors;
use actix_web::{App, HttpServer, web};
use std::sync::Arc;

use crate::routes::{health, list_scores, new_game, submit_score};
use crate::scores::ScoreStore;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.route("/api/health", web::get().to(health))
        .route("/api/games", web::post().to(new_game))
        .route("/api/scores", web::get().to(list_scores))
        .route("/api/scores", web::post().to(submit_score));
}

pub async fn run(addr: &str) -> std::io::Result<()> {
    let store = web::Data::from(Arc::new(ScoreStore::new()));
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(store.clone())
            .configure(configure)
    })
    .bind(addr)?
    .run()
    .await
}
