use actix_web::{web, HttpRequest, HttpResponse};
use crate::db::Database;
use crate::engine;
use crate::models::*;
use crate::sse::Broadcaster;
use crate::agents;
use std::sync::Arc;
use tokio::sync::broadcast;

pub struct AppState {
    pub db: Arc<Database>,
    pub broadcaster: Arc<Broadcaster>,
}

pub async fn create_game(
    data: web::Data<AppState>,
    body: web::Json<CreateGameRequest>,
) -> HttpResponse {
    if body.agents.len() < 4 || body.agents.len() > 6 {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Must select between 4 and 6 agents"
        }));
    }

    let game_id = uuid::Uuid::new_v4().to_string();
    let db = data.db.clone();
    let broadcaster = data.broadcaster.clone();
    let agents = body.agents.clone();
    let gid = game_id.clone();

    tokio::spawn(async move {
        engine::run_game(db, broadcaster, gid, agents).await;
    });

    HttpResponse::Ok().json(serde_json::json!({
        "id": game_id,
        "status": "running"
    }))
}

pub async fn get_game(
    data: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let game_id = path.into_inner();
    match data.db.get_game(&game_id) {
        Some(game) => HttpResponse::Ok().json(game),
        None => HttpResponse::NotFound().json(serde_json::json!({"error": "Game not found"})),
    }
}

pub async fn list_games(data: web::Data<AppState>) -> HttpResponse {
    let games = data.db.list_games();
    HttpResponse::Ok().json(games)
}

pub async fn list_agents() -> HttpResponse {
    HttpResponse::Ok().json(agents::get_available_agents())
}

pub async fn stream_game(
    data: web::Data<AppState>,
    path: web::Path<String>,
    _req: HttpRequest,
) -> HttpResponse {
    let game_id = path.into_inner();
    let mut rx: broadcast::Receiver<String> = data.broadcaster.subscribe(&game_id);

    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    yield Ok::<_, actix_web::Error>(actix_web::web::Bytes::from(msg));
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(stream)
}
