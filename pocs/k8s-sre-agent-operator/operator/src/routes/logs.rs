use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;
use crate::AppState;
use crate::k8s::diagnostics;

pub async fn get_logs(
    State(state): State<Arc<AppState>>,
) -> Result<String, (StatusCode, String)> {
    diagnostics::collect_all_logs(&state.k8s_client)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
