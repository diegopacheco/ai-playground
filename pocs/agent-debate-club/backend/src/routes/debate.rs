use axum::{
    extract::{Path, State},
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use chrono::Utc;
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;
use crate::AppState;
use crate::debate::engine::DebateEngine;
use crate::debate::state::DebateState;
use crate::persistence::db;
use crate::persistence::models::{CreateDebateRequest, DebateRecord};
use crate::sse::broadcaster::DebateEvent;

#[derive(serde::Serialize)]
pub struct CreateDebateResponse {
    pub id: String,
}

pub async fn create_debate(
    State(state): State<AppState>,
    Json(req): Json<CreateDebateRequest>,
) -> Json<CreateDebateResponse> {
    let id = Uuid::new_v4().to_string();

    let debate = DebateRecord {
        id: id.clone(),
        topic: req.topic.clone(),
        agent_a: req.agent_a.clone(),
        agent_b: req.agent_b.clone(),
        agent_judge: req.agent_judge.clone(),
        winner: None,
        judge_reason: None,
        duration_seconds: req.duration_seconds,
        started_at: Utc::now().to_rfc3339(),
        ended_at: None,
    };

    let _ = db::create_debate(&state.pool, &debate).await;
    let _ = state.broadcaster.create_channel(&id).await;

    let pool = state.pool.clone();
    let broadcaster = state.broadcaster.clone();
    let debate_id = id.clone();

    tokio::spawn(async move {
        let debate_state = DebateState::new(
            debate_id,
            req.topic,
            req.agent_a,
            req.agent_b,
            req.agent_judge,
            req.duration_seconds,
        );

        let engine = DebateEngine::new(pool, broadcaster);
        engine.run_debate(debate_state).await;
    });

    Json(CreateDebateResponse { id })
}

pub async fn debate_stream(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.broadcaster.subscribe(&id).await;

    let stream = async_stream::stream! {
        if let Some(rx) = rx {
            let mut stream = BroadcastStream::new(rx);
            loop {
                use tokio_stream::StreamExt;
                match stream.next().await {
                    Some(Ok(event)) => {
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        let event_type = match &event {
                            DebateEvent::AgentThinking { .. } => "agent_thinking",
                            DebateEvent::AgentMessage { .. } => "agent_message",
                            DebateEvent::DebateOver { .. } => "debate_over",
                            DebateEvent::Error { .. } => "error",
                        };
                        yield Ok(Event::default().event(event_type).data(data));
                    }
                    Some(Err(_)) => continue,
                    None => break,
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
