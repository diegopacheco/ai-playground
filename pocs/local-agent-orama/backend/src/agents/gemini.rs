use crate::agents::SharedAgentState;
use crate::models::AgentStatus;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;
use tokio::fs;

const TIMEOUT_SECS: u64 = 300;

pub async fn run(state: SharedAgentState, prompt: &str) {
    {
        let mut s = state.lock().await;
        s.status = AgentStatus::Running;
    }
    let worktree = {
        let s = state.lock().await;
        s.worktree.clone()
    };
    let log_path = worktree.join("logs.txt");
    let result = timeout(
        Duration::from_secs(TIMEOUT_SECS),
        Command::new("gemini")
            .args(["-y", prompt])
            .current_dir(&worktree)
            .output()
    ).await;
    let mut s = state.lock().await;
    match result {
        Ok(Ok(output)) => {
            let logs = format!(
                "=== STDOUT ===\n{}\n=== STDERR ===\n{}\n=== EXIT CODE: {} ===\n",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
                output.status.code().unwrap_or(-1)
            );
            let _ = fs::write(&log_path, logs).await;
            if output.status.success() {
                s.status = AgentStatus::Done;
            } else {
                s.status = AgentStatus::Error;
            }
        }
        Ok(Err(e)) => {
            let _ = fs::write(&log_path, format!("Error: {}", e)).await;
            s.status = AgentStatus::Error;
        }
        Err(_) => {
            let _ = fs::write(&log_path, "Timeout after 5 minutes").await;
            s.status = AgentStatus::Timeout;
        }
    }
}
