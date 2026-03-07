use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;
use crate::AppState;
use crate::k8s::diagnostics;
use crate::k8s::applier;
use crate::agent::runner;

pub async fn fix_deployments(
    State(state): State<Arc<AppState>>,
) -> Result<String, (StatusCode, String)> {
    let diag = diagnostics::collect_diagnostics(&state.k8s_client)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if diag.trim().is_empty() {
        return Ok("No issues detected in the cluster.".to_string());
    }

    let prompt = format!(
        "You are a Kubernetes SRE expert. Analyze the following cluster diagnostics and produce ONLY the corrected YAML manifests. \
         Each YAML document must be separated by '---'. Do not include any explanation, markdown fences, or text outside the YAML. \
         Only output valid Kubernetes YAML that can be directly applied with kubectl apply -f.\n\n\
         DIAGNOSTICS:\n{}", diag
    );

    let response = runner::run_claude(&prompt)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let yaml_content = extract_yaml(&response);

    let specs_dir = "/tmp/sre-fixes";
    std::fs::create_dir_all(specs_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let fix_path = format!("{}/fix.yaml", specs_dir);
    std::fs::write(&fix_path, &yaml_content)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let apply_result = applier::apply_file(&fix_path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(format!("Fix applied.\n\nClaude analysis:\n{}\n\nKubectl output:\n{}", response, apply_result))
}

fn extract_yaml(response: &str) -> String {
    let lines: Vec<&str> = response.lines().collect();
    let mut yaml_lines = Vec::new();
    let mut in_fence = false;

    for line in &lines {
        if line.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            yaml_lines.push(*line);
        }
    }

    if yaml_lines.is_empty() {
        return response.to_string();
    }

    yaml_lines.join("\n")
}
