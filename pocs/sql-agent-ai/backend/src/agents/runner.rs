use std::process::Stdio;
use tokio::process::Command;

pub struct AgentRunner {
    pub command: String,
    pub args: Vec<String>,
}

impl AgentRunner {
    pub fn new(command: String, args: Vec<String>) -> Self {
        Self { command, args }
    }

    pub async fn run(&self) -> Result<String, String> {
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            Command::new(&self.command)
                .args(&self.args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await;

        match result {
            Ok(Ok(output)) => {
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).to_string())
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    Err(format!("{}\n{}", stderr, stdout))
                }
            }
            Ok(Err(e)) => Err(format!("Failed to execute command: {}", e)),
            Err(_) => Err("Command timed out after 120 seconds".to_string()),
        }
    }
}
