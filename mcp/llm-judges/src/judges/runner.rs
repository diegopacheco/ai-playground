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

        let mut stdout_reader = child.stdout.take().unwrap();
        let mut stderr_reader = child.stderr.take().unwrap();

        let result = timeout(Duration::from_secs(180), async {
            let mut stdout_buf = String::new();
            let mut stderr_buf = String::new();
            let (stdout_res, stderr_res) = tokio::join!(
                stdout_reader.read_to_string(&mut stdout_buf),
                stderr_reader.read_to_string(&mut stderr_buf),
            );
            stdout_res.map_err(|e| e.to_string())?;
            stderr_res.map_err(|e| e.to_string())?;
            child.wait().await.map_err(|e| e.to_string())?;
            Ok::<(String, String), String>((stdout_buf, stderr_buf))
        })
        .await;

        match result {
            Ok(Ok((stdout_out, stderr_out))) => {
                let trimmed = stdout_out.trim().to_string();
                if trimmed.is_empty() {
                    let err_trimmed = stderr_out.trim().to_string();
                    if err_trimmed.is_empty() {
                        Err(format!("Judge {} returned empty response", self.name))
                    } else {
                        Err(format!("Judge {} failed: {}", self.name, err_trimmed))
                    }
                } else {
                    Ok(trimmed)
                }
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let _ = child.kill().await;
                Err(format!("Judge {} timed out after 180s", self.name))
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
