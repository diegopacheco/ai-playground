use tokio::process::Command;
use std::time::Duration;

pub async fn call_claude(prompt: &str) -> Result<String, String> {
    let result = tokio::time::timeout(
        Duration::from_secs(120),
        Command::new("claude")
            .arg("-p")
            .arg(prompt)
            .arg("--model")
            .arg("sonnet")
            .output()
    ).await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.trim().is_empty() {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                Err(format!("Empty response. stderr: {}", stderr))
            } else {
                Ok(stdout.trim().to_string())
            }
        }
        Ok(Err(e)) => Err(format!("Failed to run claude CLI: {}", e)),
        Err(_) => Err("Claude CLI timed out after 120s".to_string()),
    }
}
