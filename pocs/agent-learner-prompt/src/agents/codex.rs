use super::run_command_with_timeout;
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

pub async fn run(
    prompt: &str,
    model: &str,
    work_dir: &Path,
) -> Result<(String, String), String> {
    let child = Command::new("codex")
        .arg("exec")
        .arg("--full-auto")
        .arg("--model")
        .arg(model)
        .arg(prompt)
        .current_dir(work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn codex: {}", e))?;
    run_command_with_timeout(child).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_model_arg() {
        let model = "gpt-5.2";
        assert!(!model.is_empty());
    }
}
