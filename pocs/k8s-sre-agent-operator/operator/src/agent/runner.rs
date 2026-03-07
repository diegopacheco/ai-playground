use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use super::claude;

pub async fn run_claude(prompt: &str) -> Result<String, String> {
    let (cmd, args) = claude::build_command(prompt);

    let mut child = Command::new(&cmd)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn claude: {}", e))?;

    let result = timeout(Duration::from_secs(180), async {
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
                Err("Claude returned empty response".to_string())
            } else {
                Ok(trimmed)
            }
        }
        Ok(Err(e)) => Err(e),
        Err(_) => {
            let _ = child.kill().await;
            Err("Claude timed out after 180s".to_string())
        }
    }
}
