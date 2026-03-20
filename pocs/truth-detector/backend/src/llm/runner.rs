use std::time::Duration;
use tokio::process::Command;

pub async fn run_llm(prompt: &str, cli: &str, model: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let timeout_secs: u64 = std::env::var("LLM_TIMEOUT")
        .unwrap_or_else(|_| "60".to_string())
        .parse()
        .unwrap_or(60);

    let mut cmd = match cli {
        "gemini" => {
            let mut c = Command::new("gemini");
            c.args(["-y", "-p", prompt]);
            c
        }
        "copilot" => {
            let mut c = Command::new("copilot");
            c.args(["--allow-all", "--model", model, "-p", prompt]);
            c
        }
        "codex" => {
            let mut c = Command::new("codex");
            c.args(["exec", "--full-auto", "-m", model, prompt]);
            c
        }
        _ => {
            let mut c = Command::new("claude");
            c.args([
                "-p",
                prompt,
                "--model",
                model,
                "--dangerously-skip-permissions",
            ]);
            c
        }
    };

    let output = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await??;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout)
}
