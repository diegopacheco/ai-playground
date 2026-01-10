use serde_json::json;
use std::fs;

pub fn list_files(path: &str) -> String {
    let dir_path = if path.is_empty() { "." } else { path };
    match fs::read_dir(dir_path) {
        Ok(entries) => {
            let mut result = Vec::new();
            for entry in entries.flatten() {
                let file_type = if entry.path().is_dir() { "directory" } else { "file" };
                result.push(json!({
                    "name": entry.file_name().to_string_lossy(),
                    "type": file_type
                }));
            }
            serde_json::to_string_pretty(&result).unwrap_or_default()
        }
        Err(e) => format!("Error listing directory: {}", e),
    }
}
