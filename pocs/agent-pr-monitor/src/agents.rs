use std::process::Command;
use std::time::Duration;

pub fn run_llm(agent: &str, model: &str, prompt: &str) -> Result<String, String> {
    let mut cmd = match agent {
        "claude" => {
            let mut c = Command::new("claude");
            c.arg("-p").arg(prompt).arg("--model").arg(model).arg("--dangerously-skip-permissions");
            c
        }
        "gemini" => {
            let mut c = Command::new("gemini");
            c.arg("-y").arg("-p").arg(prompt);
            c
        }
        "copilot" => {
            let mut c = Command::new("copilot");
            c.arg("--allow-all").arg("--model").arg(model).arg("-p").arg(prompt);
            c
        }
        "codex" => {
            let mut c = Command::new("codex");
            c.arg("exec").arg("--full-auto").arg("-m").arg(model).arg(prompt);
            c
        }
        _ => return Err(format!("Unknown agent: {}", agent)),
    };

    let child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn {}: {}", agent, e))?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for {}: {}", agent, e))?;

    let _ = Duration::from_secs(30);

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        Err(format!("Agent {} failed: {} {}", agent, stderr, stdout))
    }
}

pub fn extract_code_block(response: &str) -> Option<String> {
    let start = response.find("```")?;
    let after_backticks = &response[start + 3..];
    let content_start = after_backticks.find('\n')? + 1;
    let rest = &after_backticks[content_start..];
    let end = rest.find("```")?;
    let code = rest[..end].trim().to_string();
    if code.is_empty() {
        None
    } else {
        Some(code)
    }
}

pub fn available_agents() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("claude", vec!["opus", "sonnet", "haiku"]),
        ("gemini", vec!["gemini-2.5-pro", "gemini-3-flash"]),
        ("copilot", vec!["claude-sonnet-4.6", "gemini-3-pro"]),
        ("codex", vec!["gpt-5.4", "gpt-5.4-mini"]),
    ]
}
