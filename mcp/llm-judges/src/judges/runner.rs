use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use super::{claude, codex, copilot, gemini};

pub struct JudgeRunner {
    pub name: String,
}

impl JudgeRunner {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_lowercase(),
        }
    }

    pub async fn run(&self, prompt: &str) -> Result<String, String> {
        let (cmd, args) = self.build_command(prompt);

        let mut child = Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {}", self.name, e))?;

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
                    Err(format!("Judge {} returned empty response", self.name))
                } else {
                    Ok(trimmed)
                }
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let _ = child.kill().await;
                Err(format!("Judge {} timed out after 120s", self.name))
            }
        }
    }

    fn build_command(&self, prompt: &str) -> (String, Vec<String>) {
        match self.name.as_str() {
            "claude" => claude::build_command(prompt),
            "codex" => codex::build_command(prompt),
            "copilot" => copilot::build_command(prompt),
            "gemini" => gemini::build_command(prompt),
            _ => ("echo".to_string(), vec!["Unknown judge".to_string()]),
        }
    }
}
