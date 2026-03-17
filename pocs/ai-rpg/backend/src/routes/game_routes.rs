use axum::{
    Router,
    extract::{Path, State},
    response::{Json, sse::{Event, Sse}},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::convert::Infallible;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use crate::AppState;
use crate::persistence::db;
use crate::game::engine;

#[derive(Deserialize)]
pub struct CreateGameRequest {
    player_name: String,
    setting: String,
}

#[derive(Serialize)]
pub struct CreateGameResponse {
    id: String,
}

#[derive(Deserialize)]
pub struct ActionRequest {
    action: String,
}

#[derive(Serialize)]
pub struct GameResponse {
    game: db::GameRow,
    messages: Vec<MessageResponse>,
    character: Option<CharacterResponse>,
}

#[derive(Serialize)]
pub struct MessageResponse {
    role: String,
    content: String,
}

#[derive(Serialize)]
pub struct CharacterResponse {
    hp: i32,
    max_hp: i32,
    level: i32,
    xp: i32,
    gold: i32,
    inventory: Vec<String>,
    location: String,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/games", post(create_game))
        .route("/games", get(list_games))
        .route("/games/{id}", get(get_game))
        .route("/games/{id}/action", post(send_action))
        .route("/games/{id}/stream", get(game_stream))
}

async fn create_game(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateGameRequest>,
) -> Json<CreateGameResponse> {
    let id = uuid::Uuid::new_v4().to_string();
    db::create_game(&state.db, &id, &req.player_name, &req.setting).await;
    state.broadcaster.create_channel(&id);

    let pool = state.db.clone();
    let broadcaster = state.broadcaster.clone();
    let game_id = id.clone();
    tokio::spawn(async move {
        engine::start_game(&pool, &broadcaster, &game_id).await;
    });

    Json(CreateGameResponse { id })
}

async fn list_games(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<db::GameRow>> {
    let games = db::get_games(&state.db).await;
    Json(games)
}

async fn get_game(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<GameResponse> {
    let game = db::get_game(&state.db, &id).await.unwrap();
    let messages = db::get_messages(&state.db, &id).await;
    let character = db::get_character(&state.db, &id).await;

    let msgs: Vec<MessageResponse> = messages.into_iter()
        .map(|(role, content)| MessageResponse { role, content })
        .collect();

    let char_resp = character.map(|c| {
        let inv: Vec<String> = serde_json::from_str(&c.inventory).unwrap_or_default();
        CharacterResponse {
            hp: c.hp,
            max_hp: c.max_hp,
            level: c.level,
            xp: c.xp,
            gold: c.gold,
            inventory: inv,
            location: c.location,
        }
    });

    Json(GameResponse {
        game,
        messages: msgs,
        character: char_resp,
    })
}

async fn send_action(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<ActionRequest>,
) -> Json<serde_json::Value> {
    if state.broadcaster.subscribe(&id).is_none() {
        state.broadcaster.create_channel(&id);
    }

    let pool = state.db.clone();
    let broadcaster = state.broadcaster.clone();
    let game_id = id.clone();
    let action = req.action.clone();
    tokio::spawn(async move {
        engine::handle_action(&pool, &broadcaster, &game_id, &action).await;
    });

    Json(serde_json::json!({"status": "ok"}))
}

async fn game_stream(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.broadcaster.subscribe(&id)
        .unwrap_or_else(|| state.broadcaster.create_channel(&id));

    let stream = BroadcastStream::new(rx)
        .filter_map(|result| {
            match result {
                Ok(event) => {
                    let event_type = match &event {
                        crate::sse::broadcaster::GameEvent::DmNarration { .. } => "dm_narration",
                        crate::sse::broadcaster::GameEvent::DmThinking => "dm_thinking",
                        crate::sse::broadcaster::GameEvent::GameOver { .. } => "game_over",
                        crate::sse::broadcaster::GameEvent::Error { .. } => "error",
                    };
                    let data = serde_json::to_string(&event).unwrap_or_default();
                    Some(Ok(Event::default().event(event_type).data(data)))
                }
                Err(_) => None,
            }
        });

    Sse::new(stream)
}
