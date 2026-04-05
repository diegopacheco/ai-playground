use std::process::Stdio;
use tokio::process::Command;

pub async fn run_agent(agent_type: &str, prompt: &str) -> Result<String, String> {
    let (cmd, args) = build_command(agent_type, prompt);

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {}", agent_type, e))?
            .wait_with_output()
    )
    .await
    .map_err(|_| format!("{} timed out after 120s", agent_type))?
    .map_err(|e| format!("{} execution failed: {}", agent_type, e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("{} failed: {}", agent_type, stderr))
    }
}

fn build_command(agent_type: &str, prompt: &str) -> (String, Vec<String>) {
    match agent_type {
        "claude" => crate::agents::claude::build(prompt),
        "gemini" => crate::agents::gemini::build(prompt),
        "copilot" => crate::agents::copilot::build(prompt),
        "codex" => crate::agents::codex::build(prompt),
        _ => ("echo".to_string(), vec![format!("Unknown agent type: {}", agent_type)]),
    }
}
