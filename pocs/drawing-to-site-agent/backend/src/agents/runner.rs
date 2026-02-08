use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use super::{claude, gemini, copilot, codex};

pub struct AgentRunner {
    pub engine: String,
}

impl AgentRunner {
    pub fn new(engine: &str) -> Self {
        Self {
            engine: engine.to_lowercase(),
        }
    }

    pub async fn run(&self, prompt: &str) -> Result<String, String> {
        let (cmd, args) = self.build_command(prompt);

        let mut child = Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {}", self.engine, e))?;

        let result = timeout(Duration::from_secs(120), async {
            let mut stdout = child.stdout.take().unwrap();
            let mut output = String::new();
            stdout.read_to_string(&mut output).await.map_err(|e| e.to_string())?;
            child.wait().await.map_err(|e| e.to_string())?;
            Ok::<String, String>(output)
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                let trimmed = output.trim().to_string();
                if trimmed.is_empty() {
                    Err(format!("Engine {} returned empty response", self.engine))
                } else {
                    Ok(trimmed)
                }
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let _ = child.kill().await;
                Err(format!("Engine {} timed out after 120s", self.engine))
            }
        }
    }

    fn build_command(&self, prompt: &str) -> (String, Vec<String>) {
        match self.engine.as_str() {
            "claude/opus" => claude::build_command(prompt, "opus"),
            "claude/sonnet" => claude::build_command(prompt, "sonnet"),
            "codex/gpt-5-2-codex" => codex::build_command(prompt),
            "gemini/gemini-3-0" => gemini::build_command(prompt),
            "copilot/sonnet" => copilot::build_command(prompt, "claude-sonnet-4.5"),
            "copilot/opus" => copilot::build_command(prompt, "claude-opus-4.6"),
            _ => ("echo".to_string(), vec!["Unknown engine".to_string()]),
        }
    }
}
