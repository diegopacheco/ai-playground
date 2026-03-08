use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;
use crate::AppState;
use crate::k8s::diagnostics;
use crate::k8s::applier;
use crate::agent::runner;
use crate::history;

#[derive(Serialize)]
pub struct FixResult {
    pub diagnostics: String,
    pub claude_response: String,
    pub kubectl_output: String,
    pub success: bool,
}

pub async fn fix_deployments(
    State(state): State<Arc<AppState>>,
) -> Result<Json<FixResult>, (StatusCode, String)> {
    let diag = diagnostics::collect_diagnostics(&state.k8s_client)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if diag.trim().is_empty() {
        return Ok(Json(FixResult {
            diagnostics: String::new(),
            claude_response: "No issues detected in the cluster.".to_string(),
            kubectl_output: String::new(),
            success: true,
        }));
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

    match applier::apply_file(&fix_path).await {
        Ok(kubectl_output) => {
            history::add_event(&state.history, "fix",
                &format!("Fixed: {}", kubectl_output),
                &format!("Diagnostics:\n{}\n\nClaude:\n{}", diag, response),
                true).await;

            Ok(Json(FixResult {
                diagnostics: diag,
                claude_response: response,
                kubectl_output,
                success: true,
            }))
        }
        Err(e) => {
            history::add_event(&state.history, "fix", &e, &diag, false).await;
            Ok(Json(FixResult {
                diagnostics: diag,
                claude_response: response,
                kubectl_output: e,
                success: false,
            }))
        }
    }
}

fn extract_yaml(response: &str) -> String {
    let mut blocks: Vec<String> = Vec::new();
    let mut current_block: Vec<&str> = Vec::new();
    let mut in_fence = false;

    for line in response.lines() {
        if line.starts_with("```") {
            if in_fence && !current_block.is_empty() {
                blocks.push(current_block.join("\n"));
                current_block.clear();
            }
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            current_block.push(line);
        }
    }

    if !blocks.is_empty() {
        return blocks.join("\n---\n");
    }

    response.to_string()
}
