use axum::{
    Router,
    extract::{Path, State},
    routing::get,
    Json,
};
use std::sync::Arc;
use crate::AppState;
use crate::persistence::models::{AgentRecord, AgentWithMessages};
use crate::persistence::db;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/agents", get(list_agents))
        .route("/api/agents/{id}", get(get_agent))
}

async fn list_agents(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<AgentRecord>> {
    let agents = db::get_all_agents(&state.pool).await.unwrap_or_default();
    Json(agents)
}

async fn get_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<Option<AgentWithMessages>> {
    let agent = db::get_agent(&state.pool, &id).await.unwrap_or(None);
    match agent {
        Some(a) => {
            let messages = db::get_messages(&state.pool, &id).await.unwrap_or_default();
            Json(Some(AgentWithMessages { agent: a, messages }))
        }
        None => Json(None),
    }
}
