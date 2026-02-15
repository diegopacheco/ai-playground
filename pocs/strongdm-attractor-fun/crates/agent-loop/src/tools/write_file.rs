use super::{Tool, ToolFuture};
use serde_json::Value;

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file at the given path"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to write" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        })
    }

    fn execute(&self, input: Value) -> ToolFuture {
        Box::pin(async move {
            let path = input["path"]
                .as_str()
                .ok_or_else(|| "missing 'path' parameter".to_string())?;
            let content = input["content"]
                .as_str()
                .ok_or_else(|| "missing 'content' parameter".to_string())?;
            if let Some(parent) = std::path::Path::new(path).parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("failed to create dirs: {}", e))?;
            }
            tokio::fs::write(path, content)
                .await
                .map_err(|e| format!("failed to write {}: {}", path, e))?;
            Ok(format!("wrote {} bytes to {}", content.len(), path))
        })
    }
}
