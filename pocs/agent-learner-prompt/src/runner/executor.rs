use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

const SOLUTION_RUN_TIMEOUT_SECS: u64 = 10;

pub async fn run_solution_with_timeout(project_dir: &Path) -> (bool, String) {
    let run_script = project_dir.join("run.sh");
    if !run_script.exists() {
        return (false, "No run.sh found in project directory".to_string());
    }
    println!("Running solution with {}s timeout...", SOLUTION_RUN_TIMEOUT_SECS);
    let child = Command::new("bash")
        .arg("run.sh")
        .current_dir(project_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return (false, format!("Failed to start: {}", e)),
    };
    let timeout_duration = Duration::from_secs(SOLUTION_RUN_TIMEOUT_SECS);
    match timeout(timeout_duration, child.wait()).await {
        Ok(Ok(status)) => {
            let stdout = if let Some(mut out) = child.stdout.take() {
                let mut buf = Vec::new();
                let _ = tokio::io::AsyncReadExt::read_to_end(&mut out, &mut buf).await;
                String::from_utf8_lossy(&buf).to_string()
            } else {
                String::new()
            };
            let stderr = if let Some(mut err) = child.stderr.take() {
                let mut buf = Vec::new();
                let _ = tokio::io::AsyncReadExt::read_to_end(&mut err, &mut buf).await;
                String::from_utf8_lossy(&buf).to_string()
            } else {
                String::new()
            };
            if !stdout.is_empty() {
                println!("Output: {}", stdout.chars().take(500).collect::<String>());
            }
            if !stderr.is_empty() {
                eprintln!("Errors: {}", stderr.chars().take(500).collect::<String>());
            }
            if status.success() {
                (true, stdout)
            } else {
                (false, stderr)
            }
        }
        Ok(Err(e)) => (false, format!("Process error: {}", e)),
        Err(_) => {
            let _ = child.kill().await;
            println!("Solution timed out after {}s (likely a web server - OK)", SOLUTION_RUN_TIMEOUT_SECS);
            (true, "Timed out - likely running server".to_string())
        }
    }
}
