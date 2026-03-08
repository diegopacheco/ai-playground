use axum::extract::State;
use axum::Json;
use std::sync::Arc;
use crate::AppState;
use crate::history::HistoryEvent;

pub async fn get_history(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<HistoryEvent>> {
    let events = state.history.lock().await.clone();
    Json(events)
}
