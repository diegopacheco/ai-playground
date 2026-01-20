use crate::agents::SharedAgentState;
use crate::models::AgentStatus;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

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
    let result = timeout(
        Duration::from_secs(TIMEOUT_SECS),
        Command::new("claude")
            .args(["-p", prompt, "--dangerously-skip-permissions"])
            .current_dir(&worktree)
            .output()
    ).await;
    let mut s = state.lock().await;
    match result {
        Ok(Ok(output)) => {
            if output.status.success() {
                s.status = AgentStatus::Done;
            } else {
                s.status = AgentStatus::Error;
            }
        }
        Ok(Err(_)) => {
            s.status = AgentStatus::Error;
        }
        Err(_) => {
            s.status = AgentStatus::Timeout;
        }
    }
}
