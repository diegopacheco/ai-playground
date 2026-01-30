pub mod claude;
pub mod codex;
pub mod copilot;
pub mod gemini;

use crate::models::{AgentResponse, ModelResult};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

fn build_analysis_prompt(prompt: &str) -> String {
    format!(
        r#"Analyze the following prompt and provide:
1. A score from 1-5 based on effectiveness
2. Specific recommendations for improvement
3. What is missing or could be clearer

Prompt to analyze:
---
{}
---

Respond ONLY with valid JSON in this exact format, nothing else:
{{"score": N, "recommendations": "your recommendations here"}}"#,
        prompt
    )
}

pub async fn execute_claude(model: &str, prompt: &str) -> ModelResult {
    let analysis_prompt = build_analysis_prompt(prompt);
    let result = timeout(Duration::from_secs(60), async {
        Command::new("claude")
            .arg("-p")
            .arg(&analysis_prompt)
            .arg("--model")
            .arg(model)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
    })
    .await;
    handle_result(result, "claude", model)
}

pub async fn execute_codex(model: &str, prompt: &str) -> ModelResult {
    let analysis_prompt = build_analysis_prompt(prompt);
    let result = timeout(Duration::from_secs(60), async {
        Command::new("codex")
            .arg("exec")
            .arg("--full-auto")
            .arg("--model")
            .arg(model)
            .arg(&analysis_prompt)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
    })
    .await;
    handle_result(result, "codex", model)
}

pub async fn execute_copilot(model: &str, prompt: &str) -> ModelResult {
    let analysis_prompt = build_analysis_prompt(prompt);
    let result = timeout(Duration::from_secs(60), async {
        Command::new("copilot")
            .arg("-p")
            .arg(&analysis_prompt)
            .arg("--model")
            .arg(model)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
    })
    .await;
    handle_result(result, "copilot", model)
}

pub async fn execute_gemini(prompt: &str) -> ModelResult {
    let analysis_prompt = build_analysis_prompt(prompt);
    let result = timeout(Duration::from_secs(60), async {
        Command::new("gemini")
            .arg("-y")
            .arg(&analysis_prompt)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
    })
    .await;
    handle_result_no_model(result, "gemini")
}

fn handle_result(
    result: Result<Result<std::process::Output, std::io::Error>, tokio::time::error::Elapsed>,
    agent_cmd: &str,
    model: &str,
) -> ModelResult {
    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_agent_response(&stdout, agent_cmd, model)
        }
        Ok(Err(e)) => ModelResult {
            model: format!("{}/{}", agent_cmd, model),
            score: None,
            recommendations: format!("Agent execution failed: {}", e),
        },
        Err(_) => ModelResult {
            model: format!("{}/{}", agent_cmd, model),
            score: None,
            recommendations: "Agent timed out after 60 seconds".to_string(),
        },
    }
}

fn handle_result_no_model(
    result: Result<Result<std::process::Output, std::io::Error>, tokio::time::error::Elapsed>,
    agent_cmd: &str,
) -> ModelResult {
    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_agent_response(&stdout, agent_cmd, "gemini-3")
        }
        Ok(Err(e)) => ModelResult {
            model: format!("{}/gemini-3", agent_cmd),
            score: None,
            recommendations: format!("Agent execution failed: {}", e),
        },
        Err(_) => ModelResult {
            model: format!("{}/gemini-3", agent_cmd),
            score: None,
            recommendations: "Agent timed out after 60 seconds".to_string(),
        },
    }
}

fn parse_agent_response(output: &str, agent_cmd: &str, model: &str) -> ModelResult {
    let model_name = format!("{}/{}", agent_cmd, model);

    let json_start = output.find('{');
    let json_end = output.rfind('}');

    if let (Some(start), Some(end)) = (json_start, json_end) {
        let json_str = &output[start..=end];
        if let Ok(response) = serde_json::from_str::<AgentResponse>(json_str) {
            return ModelResult {
                model: model_name,
                score: Some(response.score.min(5).max(1)),
                recommendations: response.recommendations,
            };
        }
    }

    ModelResult {
        model: model_name,
        score: None,
        recommendations: format!("Could not parse agent response: {}", output.chars().take(200).collect::<String>()),
    }
}
