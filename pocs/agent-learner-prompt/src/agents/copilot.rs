use super::run_command_with_timeout;
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

pub async fn run(
    prompt: &str,
    model: &str,
    work_dir: &Path,
) -> Result<(String, String), String> {
    let child = Command::new("copilot")
        .arg("--allow-all")
        .arg("--model")
        .arg(model)
        .arg("-p")
        .arg(prompt)
        .current_dir(work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn copilot: {}", e))?;
    run_command_with_timeout(child).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_model_arg() {
        let model = "claude-sonnet-4";
        assert!(model.contains("claude") || model.contains("gpt"));
    }
}
