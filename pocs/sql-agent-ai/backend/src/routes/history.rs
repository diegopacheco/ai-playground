use axum::extract::{Path, State};
use axum::Json;
use std::sync::Arc;

use crate::app_state::AppState;
use crate::persistence::db;
use crate::persistence::models::QueryRecord;

pub async fn get_queries(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<QueryRecord>> {
    Json(db::get_all_queries(&state.pool).await)
}

pub async fn get_query(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<Option<QueryRecord>> {
    Json(db::get_query(&state.pool, &id).await)
}
