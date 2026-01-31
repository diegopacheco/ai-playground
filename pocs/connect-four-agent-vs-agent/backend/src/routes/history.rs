use axum::{
    extract::{Path, State},
    Json,
};
use crate::AppState;
use crate::persistence::models::MatchRecord;

pub async fn list_matches(
    State(state): State<AppState>,
) -> Json<Vec<MatchRecord>> {
    match state.db.list_matches(100).await {
        Ok(matches) => Json(matches),
        Err(_) => Json(vec![]),
    }
}

pub async fn get_match(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Json<Option<MatchRecord>> {
    match state.db.get_match(&id).await {
        Ok(record) => Json(record),
        Err(_) => Json(None),
    }
}
