use super::{Tool, ToolFuture};
use serde_json::Value;

pub struct ApplyPatchTool;

impl Tool for ApplyPatchTool {
    fn name(&self) -> &str {
        "apply_patch"
    }

    fn description(&self) -> &str {
        "Apply a unified diff patch to files"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "patch": { "type": "string", "description": "Unified diff patch content" }
            },
            "required": ["patch"]
        })
    }

    fn execute(&self, input: Value) -> ToolFuture {
        Box::pin(async move {
            let patch = input["patch"]
                .as_str()
                .ok_or_else(|| "missing 'patch' parameter".to_string())?;
            let mut cmd = tokio::process::Command::new("patch");
            cmd.arg("-p1")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let mut child = cmd
                .spawn()
                .map_err(|e| format!("failed to spawn patch: {}", e))?;
            if let Some(mut stdin) = child.stdin.take() {
                use tokio::io::AsyncWriteExt;
                stdin
                    .write_all(patch.as_bytes())
                    .await
                    .map_err(|e| format!("failed to write patch: {}", e))?;
            }
            let output = child
                .wait_with_output()
                .await
                .map_err(|e| format!("patch failed: {}", e))?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if output.status.success() {
                Ok(format!("patch applied: {}", stdout))
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!("patch failed: {} {}", stdout, stderr))
            }
        })
    }
}
