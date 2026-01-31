use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

pub struct AgentRunner {
    pub agent_type: String,
}

impl AgentRunner {
    pub fn new(agent_type: &str) -> Self {
        Self {
            agent_type: agent_type.to_string(),
        }
    }

    pub async fn run(&self, prompt: &str) -> Result<String, String> {
        if self.agent_type == "mock" {
            return self.mock_response(prompt);
        }

        let cmd = self.get_command();
        let args = self.get_args(prompt);

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

    fn get_command(&self) -> String {
        match self.agent_type.as_str() {
            "claude" => "claude".to_string(),
            "gemini" => "gemini".to_string(),
            "copilot" => "gh".to_string(),
            "codex" => "codex".to_string(),
            _ => "claude".to_string(),
        }
    }

    fn get_args(&self, prompt: &str) -> Vec<String> {
        match self.agent_type.as_str() {
            "claude" => vec!["-p".to_string(), prompt.to_string()],
            "gemini" => vec!["-p".to_string(), prompt.to_string()],
            "copilot" => vec!["copilot".to_string(), "explain".to_string(), prompt.to_string()],
            "codex" => vec!["-p".to_string(), prompt.to_string()],
            _ => vec!["-p".to_string(), prompt.to_string()],
        }
    }
}
