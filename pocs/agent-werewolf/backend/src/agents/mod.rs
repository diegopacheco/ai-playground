pub mod claude;
pub mod gemini;
pub mod copilot;
pub mod codex;

use crate::models::{AgentInfo, AgentResponse};
use std::process::Command;
use std::time::Instant;

pub fn get_available_agents() -> Vec<AgentInfo> {
    vec![
        AgentInfo {
            name: "claude".to_string(),
            models: vec!["opus".to_string(), "sonnet".to_string(), "haiku".to_string()],
            default_model: "sonnet".to_string(),
        },
        AgentInfo {
            name: "gemini".to_string(),
            models: vec!["gemini-3.1-pro".to_string(), "gemini-3-flash".to_string(), "gemini-2.5-pro".to_string()],
            default_model: "gemini-3-flash".to_string(),
        },
        AgentInfo {
            name: "copilot".to_string(),
            models: vec!["claude-sonnet-4.6".to_string(), "claude-sonnet-4.5".to_string(), "gemini-3-pro".to_string()],
            default_model: "claude-sonnet-4.6".to_string(),
        },
        AgentInfo {
            name: "codex".to_string(),
            models: vec!["gpt-5.4".to_string(), "gpt-5.4-mini".to_string(), "gpt-5.3-codex".to_string()],
            default_model: "gpt-5.4-mini".to_string(),
        },
    ]
}

pub fn run_agent(name: &str, model: &str, prompt: &str) -> AgentResponse {
    let start = Instant::now();
    let (cmd, args) = match name {
        "claude" => claude::build_command(model, prompt),
        "gemini" => gemini::build_command(model, prompt),
        "copilot" => copilot::build_command(model, prompt),
        "codex" => codex::build_command(model, prompt),
        _ => (String::from("echo"), vec!["unknown agent".to_string()]),
    };

    let output = Command::new(&cmd)
        .args(&args)
        .output();

    let elapsed_ms = start.elapsed().as_millis() as i64;

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let combined = if stdout.trim().is_empty() { stderr } else { stdout };
            AgentResponse { output: combined, elapsed_ms }
        }
        Err(e) => AgentResponse {
            output: format!("Error running agent: {}", e),
            elapsed_ms,
        },
    }
}
