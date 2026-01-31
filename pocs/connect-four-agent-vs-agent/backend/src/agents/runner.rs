use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::io::AsyncReadExt;
use crate::agents::{claude, gemini, copilot, codex};

pub struct AgentRunner {
    agent_type: String,
}

impl AgentRunner {
    pub fn new(agent_type: &str) -> Self {
        Self {
            agent_type: agent_type.to_lowercase(),
        }
    }

    pub async fn execute(&self, prompt: &str) -> Result<u8, String> {
        let (program, args) = self.build_command(prompt);
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn {}: {}", program, e))?;
        let timeout = Duration::from_secs(60);
        let result = tokio::time::timeout(timeout, async {
            let mut stdout = child.stdout.take().unwrap();
            let mut output = String::new();
            stdout.read_to_string(&mut output).await.map_err(|e| e.to_string())?;
            let _ = child.wait().await;
            Ok::<String, String>(output)
        }).await;
        match result {
            Ok(Ok(output)) => self.parse_response(&output),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let _ = child.kill().await;
                Err("Agent timed out".to_string())
            }
        }
    }

    fn build_command(&self, prompt: &str) -> (String, Vec<String>) {
        match self.agent_type.as_str() {
            "claude" => claude::build_command(prompt),
            "gemini" => gemini::build_command(prompt),
            "copilot" => copilot::build_command(prompt),
            "codex" => codex::build_command(prompt),
            _ => ("echo".to_string(), vec!["Unknown agent".to_string()]),
        }
    }

    fn parse_response(&self, output: &str) -> Result<u8, String> {
        for c in output.chars() {
            if c.is_ascii_digit() {
                let digit = c.to_digit(10).unwrap() as u8;
                if digit <= 6 {
                    return Ok(digit);
                }
            }
        }
        Err(format!("Could not parse valid column from output: {}", output))
    }
}

pub fn list_available_agents() -> Vec<String> {
    vec![
        "claude".to_string(),
        "gemini".to_string(),
        "copilot".to_string(),
        "codex".to_string(),
    ]
}
