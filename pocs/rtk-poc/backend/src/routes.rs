use actix_web::{HttpResponse, Responder, web};
use serde_json::json;

use crate::game::build_deck;
use crate::scores::{ScoreInput, ScoreStore, validate};

pub async fn health() -> impl Responder {
    HttpResponse::Ok().json(json!({ "status": "ok" }))
}

pub async fn new_game() -> impl Responder {
    HttpResponse::Ok().json(json!({ "deck": build_deck() }))
}

pub async fn list_scores(store: web::Data<ScoreStore>) -> impl Responder {
    HttpResponse::Ok().json(json!({ "scores": store.list() }))
}

pub async fn submit_score(
    store: web::Data<ScoreStore>,
    input: web::Json<ScoreInput>,
) -> impl Responder {
    match validate(&input) {
        Ok(score) => {
            store.add(score.clone());
            HttpResponse::Created().json(score)
        }
        Err(e) => {
            let message = match e {
                crate::scores::ScoreError::EmptyName => "name must not be empty",
                crate::scores::ScoreError::ZeroMoves => "moves must be greater than 0",
                crate::scores::ScoreError::ZeroSeconds => "seconds must be greater than 0",
            };
            HttpResponse::BadRequest().json(json!({ "error": message }))
        }
    }
}
