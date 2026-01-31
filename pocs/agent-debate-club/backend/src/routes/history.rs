use axum::{
    extract::{Path, State},
    Json,
};
use crate::AppState;
use crate::persistence::db;
use crate::persistence::models::{AgentInfo, DebateRecord, DebateResponse};
use crate::agents::get_available_agents;

pub async fn get_debates(State(state): State<AppState>) -> Json<Vec<DebateRecord>> {
    let debates = db::get_all_debates(&state.pool).await.unwrap_or_default();
    Json(debates)
}

pub async fn get_debate(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Json<Option<DebateResponse>> {
    let debate = db::get_debate(&state.pool, &id).await.ok().flatten();

    match debate {
        Some(d) => {
            let messages = db::get_messages(&state.pool, &id).await.unwrap_or_default();
            Json(Some(DebateResponse {
                id: d.id,
                topic: d.topic,
                agent_a: d.agent_a,
                agent_b: d.agent_b,
                agent_judge: d.agent_judge,
                winner: d.winner,
                judge_reason: d.judge_reason,
                duration_seconds: d.duration_seconds,
                started_at: d.started_at,
                ended_at: d.ended_at,
                messages,
            }))
        }
        None => Json(None),
    }
}

pub async fn get_agents() -> Json<Vec<AgentInfo>> {
    let agents = get_available_agents()
        .into_iter()
        .map(|(id, name)| AgentInfo { id, name })
        .collect();
    Json(agents)
}
