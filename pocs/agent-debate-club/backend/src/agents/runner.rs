use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use super::{claude, gemini, copilot, codex};

pub struct AgentRunner {
    pub agent_type: String,
}

impl AgentRunner {
    pub fn new(agent_type: &str) -> Self {
        Self {
            agent_type: agent_type.to_lowercase(),
        }
    }

    pub async fn run(&self, prompt: &str) -> Result<String, String> {
        if self.agent_type == "mock" {
            return self.mock_response(prompt);
        }

        let (cmd, args) = self.build_command(prompt);

        let mut child = Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {}", self.agent_type, e))?;

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
                    Err(format!("Agent {} returned empty response", self.agent_type))
                } else {
                    Ok(trimmed)
                }
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let _ = child.kill().await;
                Err(format!("Agent {} timed out after 120s", self.agent_type))
            }
        }
    }

    fn mock_response(&self, prompt: &str) -> Result<String, String> {
        let responses = [
            "[ATTACK] This position fails to account for fundamental technical limitations that make it impractical in real-world scenarios.",
            "[DEFENSE] The evidence strongly supports this approach, as demonstrated by widespread industry adoption and measurable improvements in productivity.",
            "[ATTACK] While the previous argument sounds compelling, it overlooks critical security and maintenance concerns that outweigh any short-term benefits.",
            "[DEFENSE] Historical data and current trends clearly validate this methodology as the superior choice for modern development workflows.",
        ];
        let idx = prompt.len() % responses.len();
        Ok(responses[idx].to_string())
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
}

pub fn list_available_agents() -> Vec<String> {
    vec![
        "claude".to_string(),
        "gemini".to_string(),
        "copilot".to_string(),
        "codex".to_string(),
    ]
}
