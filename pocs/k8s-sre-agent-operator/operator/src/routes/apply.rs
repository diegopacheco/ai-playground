use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;
use crate::AppState;
use crate::k8s::applier;
use crate::history;

pub async fn apply_yaml(
    State(state): State<Arc<AppState>>,
    body: String,
) -> Result<String, (StatusCode, String)> {
    if body.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty YAML body".to_string()));
    }

    let fix_dir = "/tmp/sre-fixes";
    std::fs::create_dir_all(fix_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let fix_path = format!("{}/fix.yaml", fix_dir);
    std::fs::write(&fix_path, &body)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    match applier::apply_file(&fix_path).await {
        Ok(result) => {
            history::add_event(&state.history, "apply", &result, &body, true).await;
            Ok(result)
        }
        Err(e) => {
            history::add_event(&state.history, "apply", &e, &body, false).await;
            Err((StatusCode::INTERNAL_SERVER_ERROR, e))
        }
    }
}
