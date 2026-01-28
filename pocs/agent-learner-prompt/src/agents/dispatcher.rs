use std::path::Path;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::time::timeout;

use super::claude;
use super::codex;
use super::copilot;
use super::gemini;

const AGENT_TIMEOUT_SECS: u64 = 300;

pub const COPILOT_MODELS: &[&str] = &[
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-sonnet-4",
    "gemini-3-pro",
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5.1",
    "gpt-5",
    "gpt-5.1-codex-mini",
    "gpt-5-mini",
    "gpt-4.1",
];

pub const CLAUDE_MODELS: &[&str] = &["opus", "sonnet", "haiku"];
pub const CODEX_MODELS: &[&str] = &[
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
];
pub const GEMINI_MODELS: &[&str] = &[
    "auto-gemini-3",
    "gemini-3-pro",
    "gemini-3-flash",
    "auto-gemini-2.5",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
];

pub struct AgentResult {
    pub success: bool,
    pub output: String,
    pub error: String,
}

pub async fn run_agent(
    agent: &str,
    prompt: &str,
    model: &str,
    work_dir: &Path,
) -> AgentResult {
    let result = match agent {
        "claude" => claude::run(prompt, model, work_dir).await,
        "codex" => codex::run(prompt, model, work_dir).await,
        "copilot" => copilot::run(prompt, model, work_dir).await,
        "gemini" => gemini::run(prompt, model, work_dir).await,
        _ => Err(format!("Unknown agent: {}", agent)),
    };
    match result {
        Ok((output, error)) => AgentResult {
            success: true,
            output,
            error,
        },
        Err(e) => AgentResult {
            success: false,
            output: String::new(),
            error: e,
        },
    }
}

pub fn get_default_model(agent: &str) -> &'static str {
    match agent {
        "claude" => "sonnet",
        "codex" => "gpt-5.2-codex",
        "copilot" => "claude-sonnet-4.5",
        "gemini" => "auto-gemini-3",
        _ => "sonnet",
    }
}

pub fn is_valid_agent(agent: &str) -> bool {
    matches!(agent, "claude" | "codex" | "copilot" | "gemini")
}

pub fn get_models_for_agent(agent: &str) -> &'static [&'static str] {
    match agent {
        "claude" => CLAUDE_MODELS,
        "codex" => CODEX_MODELS,
        "copilot" => COPILOT_MODELS,
        "gemini" => GEMINI_MODELS,
        _ => &[],
    }
}

pub async fn run_command_with_timeout(
    mut child: tokio::process::Child,
) -> Result<(String, String), String> {
    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;
    let stdout_reader = BufReader::new(stdout);
    let stderr_reader = BufReader::new(stderr);
    let stdout_handle = tokio::spawn(async move {
        let mut lines = stdout_reader.lines();
        let mut output = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            println!("{}", line);
            output.push_str(&line);
            output.push('\n');
        }
        output
    });
    let stderr_handle = tokio::spawn(async move {
        let mut lines = stderr_reader.lines();
        let mut output = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            eprintln!("ERR: {}", line);
            output.push_str(&line);
            output.push('\n');
        }
        output
    });
    let timeout_duration = Duration::from_secs(AGENT_TIMEOUT_SECS);
    match timeout(timeout_duration, child.wait()).await {
        Ok(Ok(status)) => {
            let stdout_output = stdout_handle.await.unwrap_or_default();
            let stderr_output = stderr_handle.await.unwrap_or_default();
            if status.success() {
                Ok((stdout_output, stderr_output))
            } else {
                Err(format!("Agent exited with status: {}", status))
            }
        }
        Ok(Err(e)) => Err(format!("Process error: {}", e)),
        Err(_) => {
            let _ = child.kill().await;
            Err("Agent timed out".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_agent() {
        assert!(is_valid_agent("claude"));
        assert!(is_valid_agent("codex"));
        assert!(is_valid_agent("copilot"));
        assert!(is_valid_agent("gemini"));
        assert!(!is_valid_agent("unknown"));
    }

    #[test]
    fn test_get_default_model() {
        assert_eq!(get_default_model("claude"), "sonnet");
        assert_eq!(get_default_model("codex"), "gpt-5.2-codex");
        assert_eq!(get_default_model("copilot"), "claude-sonnet-4.5");
        assert_eq!(get_default_model("gemini"), "auto-gemini-3");
    }

    #[test]
    fn test_get_models_for_agent() {
        assert_eq!(get_models_for_agent("copilot").len(), 14);
        assert!(get_models_for_agent("copilot").contains(&"claude-sonnet-4.5"));
        assert_eq!(get_models_for_agent("claude").len(), 3);
        assert_eq!(get_models_for_agent("codex").len(), 4);
        assert_eq!(get_models_for_agent("gemini").len(), 6);
    }
}
