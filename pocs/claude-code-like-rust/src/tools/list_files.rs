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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_list_files_current_directory() {
        let result = list_files("");
        assert!(result.contains("["));
        assert!(result.contains("]"));
    }

    #[test]
    fn test_list_files_explicit_dot() {
        let result = list_files(".");
        assert!(result.contains("["));
        assert!(result.contains("]"));
    }

    #[test]
    fn test_list_files_nonexistent_directory() {
        let result = list_files("/nonexistent/directory/path");
        assert!(result.starts_with("Error listing directory:"));
    }

    #[test]
    fn test_list_files_with_files_and_dirs() {
        let temp_dir = env::temp_dir().join("test_list_files_dir");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        fs::write(temp_dir.join("test.txt"), "content").unwrap();
        fs::create_dir(temp_dir.join("subdir")).unwrap();
        let result = list_files(temp_dir.to_str().unwrap());
        assert!(result.contains("test.txt"));
        assert!(result.contains("subdir"));
        assert!(result.contains("\"type\": \"file\""));
        assert!(result.contains("\"type\": \"directory\""));
        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_list_files_empty_directory() {
        let temp_dir = env::temp_dir().join("test_list_files_empty");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        let result = list_files(temp_dir.to_str().unwrap());
        assert_eq!(result, "[]");
        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_list_files_returns_json_array() {
        let temp_dir = env::temp_dir().join("test_list_files_json");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        fs::write(temp_dir.join("file1.txt"), "").unwrap();
        fs::write(temp_dir.join("file2.txt"), "").unwrap();
        let result = list_files(temp_dir.to_str().unwrap());
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
