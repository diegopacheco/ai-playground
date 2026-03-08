use axum::http::StatusCode;
use crate::k8s::applier;

pub async fn apply_yaml(body: String) -> Result<String, (StatusCode, String)> {
    if body.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty YAML body".to_string()));
    }

    let fix_dir = "/tmp/sre-fixes";
    std::fs::create_dir_all(fix_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let fix_path = format!("{}/fix.yaml", fix_dir);
    std::fs::write(&fix_path, &body)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let result = applier::apply_file(&fix_path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(result)
}
