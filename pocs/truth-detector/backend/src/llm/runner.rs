use std::time::Duration;
use tokio::process::Command;

pub async fn run_llm(prompt: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let cli = std::env::var("LLM_CLI").unwrap_or_else(|_| "claude".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "sonnet".to_string());
    let timeout_secs: u64 = std::env::var("LLM_TIMEOUT")
        .unwrap_or_else(|_| "30".to_string())
        .parse()
        .unwrap_or(30);

    let mut cmd = match cli.as_str() {
        "gemini" => {
            let mut c = Command::new("gemini");
            c.args(["-y", "-p", prompt]);
            c
        }
        "copilot" => {
            let mut c = Command::new("copilot");
            c.args(["--allow-all", "--model", &model, "-p", prompt]);
            c
        }
        _ => {
            let mut c = Command::new("claude");
            c.args([
                "-p",
                prompt,
                "--model",
                &model,
                "--dangerously-skip-permissions",
            ]);
            c
        }
    };

    let output = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await??;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout)
}
