use std::process::Command;
use std::fs;

pub fn run_llm(agent: &str, model: &str, prompt: &str) -> Result<String, String> {
    let prompt = prompt.replace('\0', "");
    let tmp_file = format!("/tmp/agent-pr-prompt-{}.txt", std::process::id());
    fs::write(&tmp_file, &prompt)
        .map_err(|e| format!("Failed to write prompt file: {}", e))?;

    let mut cmd = match agent {
        "claude" => {
            let mut c = Command::new("sh");
            c.arg("-c").arg(format!("cat '{}' | claude -p --model {} --dangerously-skip-permissions", tmp_file, model));
            c
        }
        "gemini" => {
            let mut c = Command::new("sh");
            c.arg("-c").arg(format!("cat '{}' | gemini -y -p", tmp_file));
            c
        }
        "copilot" => {
            let mut c = Command::new("sh");
            c.arg("-c").arg(format!("cat '{}' | copilot --allow-all --model {} -p", tmp_file, model));
            c
        }
        "codex" => {
            let mut c = Command::new("sh");
            c.arg("-c").arg(format!("cat '{}' | codex exec --full-auto -m {}", tmp_file, model));
            c
        }
        _ => {
            let _ = fs::remove_file(&tmp_file);
            return Err(format!("Unknown agent: {}", agent));
        }
    };

    let output = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| {
            let _ = fs::remove_file(&tmp_file);
            format!("Failed to spawn {}: {}", agent, e)
        })?;

    let _ = fs::remove_file(&tmp_file);

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

pub fn extract_all_code_blocks(response: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut remaining = response;
    while let Some(start) = remaining.find("```") {
        let after = &remaining[start + 3..];
        let content_start = match after.find('\n') {
            Some(i) => i + 1,
            None => break,
        };
        let rest = &after[content_start..];
        let end = match rest.find("```") {
            Some(i) => i,
            None => break,
        };
        let code = rest[..end].trim().to_string();
        if !code.is_empty() {
            blocks.push(code);
        }
        remaining = &rest[end + 3..];
    }
    blocks
}

pub fn available_agents() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("claude", vec!["opus", "sonnet", "haiku"]),
        ("gemini", vec!["gemini-2.5-pro", "gemini-3-flash"]),
        ("copilot", vec!["claude-sonnet-4.6", "gemini-3-pro"]),
        ("codex", vec!["gpt-5.4", "gpt-5.4-mini"]),
    ]
}
