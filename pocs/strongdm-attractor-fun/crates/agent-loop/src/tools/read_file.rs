use super::{Tool, ToolFuture};
use serde_json::Value;

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to read" }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, input: Value) -> ToolFuture {
        Box::pin(async move {
            let path = input["path"]
                .as_str()
                .ok_or_else(|| "missing 'path' parameter".to_string())?;
            tokio::fs::read_to_string(path)
                .await
                .map_err(|e| format!("failed to read {}: {}", path, e))
        })
    }
}
