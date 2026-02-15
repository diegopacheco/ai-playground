use super::{Tool, ToolFuture};
use serde_json::Value;

pub struct EditFileTool;

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing old_string with new_string"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to edit" },
                "old_string": { "type": "string", "description": "String to find and replace" },
                "new_string": { "type": "string", "description": "Replacement string" }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    fn execute(&self, input: Value) -> ToolFuture {
        Box::pin(async move {
            let path = input["path"]
                .as_str()
                .ok_or_else(|| "missing 'path' parameter".to_string())?;
            let old = input["old_string"]
                .as_str()
                .ok_or_else(|| "missing 'old_string' parameter".to_string())?;
            let new = input["new_string"]
                .as_str()
                .ok_or_else(|| "missing 'new_string' parameter".to_string())?;
            let content = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| format!("failed to read {}: {}", path, e))?;
            if !content.contains(old) {
                return Err(format!("old_string not found in {}", path));
            }
            let updated = content.replacen(old, new, 1);
            tokio::fs::write(path, &updated)
                .await
                .map_err(|e| format!("failed to write {}: {}", path, e))?;
            Ok(format!("edited {}", path))
        })
    }
}
