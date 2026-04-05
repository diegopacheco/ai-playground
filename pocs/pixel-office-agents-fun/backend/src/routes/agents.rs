use axum::{
    Router,
    extract::{Path, State},
    response::sse::{Event, Sse},
    routing::{get, post, delete},
    Json,
};
use futures::stream::Stream;
use std::sync::Arc;
use std::convert::Infallible;
use crate::AppState;
use crate::persistence::models::{CreateAgentRequest, AgentRecord, MessageRecord};
use crate::persistence::db;
use crate::agents::runner;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/agents/spawn", post(spawn_agent))
        .route("/api/agents/clear", delete(clear_agents))
        .route("/api/agents/{id}/stream", get(stream_agent))
        .route("/api/agents/{id}/chat", post(chat_with_agent))
        .route("/api/agents/{id}", delete(stop_agent))
        .route("/api/agent-types", get(get_agent_types))
}

async fn spawn_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateAgentRequest>,
) -> Json<AgentRecord> {
    let id = uuid::Uuid::new_v4().to_string();
    let desk_index = db::get_next_desk_index(&state.pool).await.unwrap_or(0);
    let now = chrono::Utc::now().to_rfc3339();

    let agent = AgentRecord {
        id: id.clone(),
        name: req.name.clone(),
        agent_type: req.agent_type.clone(),
        task: req.task.clone(),
        status: "spawning".to_string(),
        desk_index,
        created_at: now,
        completed_at: None,
    };

    db::create_agent(&state.pool, &agent).await.unwrap();
    let _rx = state.broadcaster.create_channel(&id);

    let agent_clone = agent.clone();
    let state_clone = state.clone();

    tokio::spawn(async move {
        let agent_id = agent_clone.id.clone();
        let broadcaster = &state_clone.broadcaster;

        broadcaster.broadcast(&agent_id, serde_json::json!({
            "type": "agent_status",
            "agent_id": agent_id,
            "status": "thinking"
        }).to_string());

        db::update_agent_status(&state_clone.pool, &agent_id, "thinking", None).await.ok();

        broadcaster.broadcast(&agent_id, serde_json::json!({
            "type": "agent_status",
            "agent_id": agent_id,
            "status": "working"
        }).to_string());

        db::update_agent_status(&state_clone.pool, &agent_id, "working", None).await.ok();

        match runner::run_agent(&agent_clone.agent_type, &agent_clone.task).await {
            Ok(result) => {
                let msg = MessageRecord {
                    id: uuid::Uuid::new_v4().to_string(),
                    agent_id: agent_id.clone(),
                    content: result.clone(),
                    role: "assistant".to_string(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                };
                db::save_message(&state_clone.pool, &msg).await.ok();

                broadcaster.broadcast(&agent_id, serde_json::json!({
                    "type": "agent_message",
                    "agent_id": agent_id,
                    "content": result
                }).to_string());

                let completed_at = chrono::Utc::now().to_rfc3339();
                db::update_agent_status(&state_clone.pool, &agent_id, "done", Some(&completed_at)).await.ok();

                broadcaster.broadcast(&agent_id, serde_json::json!({
                    "type": "agent_done",
                    "agent_id": agent_id,
                    "status": "done"
                }).to_string());
            }
            Err(err) => {
                let completed_at = chrono::Utc::now().to_rfc3339();
                db::update_agent_status(&state_clone.pool, &agent_id, "error", Some(&completed_at)).await.ok();

                broadcaster.broadcast(&agent_id, serde_json::json!({
                    "type": "agent_error",
                    "agent_id": agent_id,
                    "message": err
                }).to_string());
            }
        }
    });

    Json(agent)
}

async fn stream_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.broadcaster.subscribe(&id);

    let stream = async_stream::stream! {
        if let Some(mut rx) = rx {
            loop {
                match rx.recv().await {
                    Ok(msg) => {
                        let event_type = serde_json::from_str::<serde_json::Value>(&msg)
                            .ok()
                            .and_then(|v| v.get("type").and_then(|t| t.as_str().map(String::from)))
                            .unwrap_or_else(|| "message".to_string());
                        yield Ok(Event::default().event(event_type).data(msg));
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        }
    };

    Sse::new(stream)
}

async fn stop_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<serde_json::Value> {
    let completed_at = chrono::Utc::now().to_rfc3339();
    db::update_agent_status(&state.pool, &id, "stopped", Some(&completed_at)).await.ok();
    state.broadcaster.broadcast(&id, serde_json::json!({
        "type": "agent_done",
        "agent_id": id,
        "status": "stopped"
    }).to_string());
    state.broadcaster.remove_channel(&id);
    Json(serde_json::json!({ "success": true }))
}

async fn clear_agents(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    db::clear_all_agents(&state.pool).await.ok();
    Json(serde_json::json!({ "success": true }))
}

#[derive(serde::Deserialize)]
struct ChatRequest {
    message: String,
}

async fn chat_with_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<ChatRequest>,
) -> Json<serde_json::Value> {
    let agent = db::get_agent(&state.pool, &id).await.unwrap_or(None);
    let agent = match agent {
        Some(a) => a,
        None => return Json(serde_json::json!({ "error": "Agent not found" })),
    };

    let user_msg = MessageRecord {
        id: uuid::Uuid::new_v4().to_string(),
        agent_id: id.clone(),
        content: req.message.clone(),
        role: "user".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    db::save_message(&state.pool, &user_msg).await.ok();

    let history = db::get_messages(&state.pool, &id).await.unwrap_or_default();
    let mut prompt = format!("You are {}. Previous conversation:\n", agent.name);
    for msg in &history {
        let role_label = if msg.role == "user" { "User" } else { "Assistant" };
        prompt.push_str(&format!("{}: {}\n", role_label, msg.content));
    }
    prompt.push_str(&format!("\nUser: {}\nRespond concisely:", req.message));

    match runner::run_agent(&agent.agent_type, &prompt).await {
        Ok(result) => {
            let assistant_msg = MessageRecord {
                id: uuid::Uuid::new_v4().to_string(),
                agent_id: id.clone(),
                content: result.clone(),
                role: "assistant".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
            };
            db::save_message(&state.pool, &assistant_msg).await.ok();
            Json(serde_json::json!({ "role": "assistant", "content": result }))
        }
        Err(err) => {
            Json(serde_json::json!({ "role": "assistant", "content": format!("Error: {}", err) }))
        }
    }
}

async fn get_agent_types() -> Json<Vec<serde_json::Value>> {
    Json(vec![
        serde_json::json!({ "id": "claude", "name": "Claude", "model": "opus", "color": "#D97706" }),
        serde_json::json!({ "id": "gemini", "name": "Gemini", "model": "gemini-3.0", "color": "#4285F4" }),
        serde_json::json!({ "id": "copilot", "name": "Copilot", "model": "claude-sonnet-4", "color": "#6366F1" }),
        serde_json::json!({ "id": "codex", "name": "Codex", "model": "gpt-5.4", "color": "#10B981" }),
    ])
}
