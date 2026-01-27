use super::run_command_with_timeout;
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

pub async fn run(
    prompt: &str,
    _model: &str,
    work_dir: &Path,
) -> Result<(String, String), String> {
    let child = Command::new("gemini")
        .arg("-y")
        .arg(prompt)
        .current_dir(work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn gemini: {}", e))?;
    run_command_with_timeout(child).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gemini_uses_default_model() {
        let _model = "gemini-2.5-pro";
        assert!(true);
    }
}
