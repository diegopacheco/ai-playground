use axum::{
    extract::{Path, State},
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;
use crate::AppState;
use crate::game::engine::GameEngine;
use crate::game::state::GameState;
use crate::sse::broadcaster::GameEvent;

#[derive(Deserialize)]
pub struct StartGameRequest {
    pub agent_a: String,
    pub agent_b: String,
}

#[derive(Serialize)]
pub struct StartGameResponse {
    pub game_id: String,
}

pub async fn start_game(
    State(state): State<AppState>,
    Json(req): Json<StartGameRequest>,
) -> Json<StartGameResponse> {
    let game_id = uuid::Uuid::new_v4().to_string();
    let game_state = GameState::new(game_id.clone(), req.agent_a, req.agent_b);
    {
        let mut broadcaster = state.broadcaster.write().await;
        broadcaster.create_channel(&game_id);
    }
    let db = state.db.clone();
    let broadcaster = state.broadcaster.clone();
    let game_id_clone = game_id.clone();
    tokio::spawn(async move {
        let mut engine = GameEngine::new(game_state, db, broadcaster.clone());
        engine.run().await;
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        let mut broadcaster = broadcaster.write().await;
        broadcaster.remove_channel(&game_id_clone);
    });
    Json(StartGameResponse { game_id })
}

pub async fn game_stream(
    State(state): State<AppState>,
    Path(game_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = {
        let broadcaster = state.broadcaster.read().await;
        broadcaster.subscribe(&game_id)
    };
    let stream = match rx {
        Some(rx) => {
            let stream = BroadcastStream::new(rx);
            stream
                .filter_map(|result: Result<GameEvent, BroadcastStreamRecvError>| async move {
                    result.ok().map(|event| {
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        Ok(Event::default().event(event_type(&event)).data(data))
                    })
                })
                .boxed()
        }
        None => {
            futures::stream::once(async {
                Ok(Event::default().event("error").data("Game not found"))
            }).boxed()
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn event_type(event: &GameEvent) -> &'static str {
    match event {
        GameEvent::BoardUpdate { .. } => "board_update",
        GameEvent::AgentThinking { .. } => "agent_thinking",
        GameEvent::AgentMoved { .. } => "agent_moved",
        GameEvent::GameOver { .. } => "game_over",
        GameEvent::Error { .. } => "error",
    }
}
