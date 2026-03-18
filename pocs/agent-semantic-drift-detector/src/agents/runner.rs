use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

pub async fn call_llm(prompt: &str) -> Result<String, String> {
    let result = timeout(
        Duration::from_secs(120),
        Command::new("claude")
            .arg("-p")
            .arg(prompt)
            .arg("--model")
            .arg("sonnet")
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
                Ok(text)
            } else {
                let err = String::from_utf8_lossy(&output.stderr).to_string();
                Err(format!("LLM call failed: {}", err))
            }
        }
        Ok(Err(e)) => Err(format!("Process error: {}", e)),
        Err(_) => Err("LLM call timed out after 120s".to_string()),
    }
}
