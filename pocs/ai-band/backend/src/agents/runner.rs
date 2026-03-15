use tokio::process::Command;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::time::{timeout, Duration};
use crate::agents::claude;

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
        let (cmd, args) = claude::build_command(prompt);

        let mut child = Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {}", self.agent_type, e))?;

        let result = timeout(Duration::from_secs(120), async {
            let mut stdout = child.stdout.take().unwrap();
            let mut output = String::new();
            stdout
                .read_to_string(&mut output)
                .await
                .map_err(|e| e.to_string())?;
            child.wait().await.map_err(|e| e.to_string())?;
            Ok::<String, String>(output)
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                let trimmed = output.trim().to_string();
                if trimmed.is_empty() {
                    Err(format!("{} returned empty response", self.agent_type))
                } else {
                    Ok(trimmed)
                }
            }
            Ok(Err(e)) => Err(format!("{} error: {}", self.agent_type, e)),
            Err(_) => {
                let _ = child.kill().await;
                Err(format!("{} timed out after 120s", self.agent_type))
            }
        }
    }
}
