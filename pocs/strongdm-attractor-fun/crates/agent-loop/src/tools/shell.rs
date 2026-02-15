use super::{Tool, ToolFuture};
use serde_json::Value;

pub struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return stdout/stderr"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to execute" },
                "cwd": { "type": "string", "description": "Working directory" }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, input: Value) -> ToolFuture {
        Box::pin(async move {
            let command = input["command"]
                .as_str()
                .ok_or_else(|| "missing 'command' parameter".to_string())?;
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(command);
            if let Some(cwd) = input["cwd"].as_str() {
                cmd.current_dir(cwd);
            }
            let output = cmd
                .output()
                .await
                .map_err(|e| format!("failed to execute: {}", e))?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str("STDERR: ");
                result.push_str(&stderr);
            }
            if output.status.success() {
                Ok(result)
            } else {
                Err(format!(
                    "exit code {}: {}",
                    output.status.code().unwrap_or(-1),
                    result
                ))
            }
        })
    }
}
