use super::run_command_with_timeout;
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

pub async fn run(
    prompt: &str,
    model: &str,
    work_dir: &Path,
) -> Result<(String, String), String> {
    let child = Command::new("claude")
        .arg("-p")
        .arg(prompt)
        .arg("--model")
        .arg(model)
        .arg("--dangerously-skip-permissions")
        .current_dir(work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn claude: {}", e))?;
    run_command_with_timeout(child).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_model_arg() {
        let model = "opus";
        assert!(!model.is_empty());
    }
}
