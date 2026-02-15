use super::{Tool, ToolFuture};
use serde_json::Value;

pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files recursively"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regex pattern to search" },
                "path": { "type": "string", "description": "Directory to search in" },
                "include": { "type": "string", "description": "File glob pattern" }
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
            let mut cmd = tokio::process::Command::new("grep");
            cmd.arg("-rn").arg(pattern).arg(path);
            if let Some(include) = input["include"].as_str() {
                cmd.arg("--include").arg(include);
            }
            let output = cmd
                .output()
                .await
                .map_err(|e| format!("grep failed: {}", e))?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.is_empty() {
                Ok("no matches found".to_string())
            } else {
                Ok(stdout)
            }
        })
    }
}
