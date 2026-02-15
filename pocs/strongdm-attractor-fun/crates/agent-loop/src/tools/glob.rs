use super::{Tool, ToolFuture};
use serde_json::Value;

pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern" },
                "path": { "type": "string", "description": "Base directory" }
            },
            "required": ["pattern"]
        })
    }

    fn execute(&self, input: Value) -> ToolFuture {
        Box::pin(async move {
            let pattern = input["pattern"]
                .as_str()
                .ok_or_else(|| "missing 'pattern' parameter".to_string())?;
            let path = input["path"].as_str().unwrap_or(".");
            let mut cmd = tokio::process::Command::new("find");
            cmd.arg(path).arg("-name").arg(pattern).arg("-type").arg("f");
            let output = cmd
                .output()
                .await
                .map_err(|e| format!("glob failed: {}", e))?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.is_empty() {
                Ok("no files found".to_string())
            } else {
                Ok(stdout)
            }
        })
    }
}
