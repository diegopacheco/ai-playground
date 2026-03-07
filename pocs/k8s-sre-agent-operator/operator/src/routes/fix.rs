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

    let mut yaml_lines: Vec<&str> = Vec::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("apiVersion:")
            || trimmed.starts_with("kind:")
            || trimmed.starts_with("metadata:")
            || trimmed.starts_with("spec:")
            || trimmed.starts_with("---")
            || trimmed.starts_with("- ")
            || trimmed.starts_with("name:")
            || trimmed.starts_with("namespace:")
            || trimmed.starts_with("labels:")
            || trimmed.starts_with("containers:")
            || trimmed.starts_with("image:")
            || trimmed.starts_with("ports:")
            || trimmed.starts_with("replicas:")
            || trimmed.starts_with("selector:")
            || trimmed.starts_with("template:")
            || trimmed.starts_with("matchLabels:")
            || trimmed.starts_with("app:")
            || trimmed.starts_with("env:")
            || trimmed.starts_with("containerPort:")
            || line.starts_with("  ")
        {
            yaml_lines.push(line);
        }
    }

    yaml_lines.join("\n")
}
