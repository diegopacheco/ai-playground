use tokio::process::Command;
use tokio::io::AsyncReadExt;
use std::process::Stdio;
use std::time::Duration;

pub struct AgentRunner;

impl AgentRunner {
    pub async fn call_llm(prompt: &str) -> Result<String, String> {
        let mut child = Command::new("claude")
            .args(&["-p", prompt, "--model", "sonnet", "--dangerously-skip-permissions"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn claude: {}", e))?;

        let result = tokio::time::timeout(
            Duration::from_secs(120),
            async {
                let mut stdout = child.stdout.take().unwrap();
                let mut output = String::new();
                stdout.read_to_string(&mut output).await.map_err(|e| format!("Read error: {}", e))?;
                child.wait().await.map_err(|e| format!("Wait error: {}", e))?;
                Ok::<String, String>(output)
            }
        ).await.map_err(|_| "LLM call timed out after 120s".to_string())?;

        let output = result?;
        let trimmed = output.trim().to_string();
        if trimmed.is_empty() {
            return Err("Empty response from LLM".to_string());
        }
        Ok(trimmed)
    }
}
