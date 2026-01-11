use std::fs;
use std::path::Path;

pub fn edit_file(path: &str, content: &str) -> String {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                return format!("Error creating directories: {}", e);
            }
        }
    }
    match fs::write(path, content) {
        Ok(_) => format!("File '{}' written successfully", path),
        Err(e) => format!("Error writing file: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_edit_file_create_new_file() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_edit_new.txt");
        let _ = fs::remove_file(&test_file);
        let result = edit_file(test_file.to_str().unwrap(), "New content");
        assert!(result.contains("written successfully"));
        let content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "New content");
        fs::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_edit_file_overwrite_existing() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_edit_overwrite.txt");
        fs::write(&test_file, "Old content").unwrap();
        let result = edit_file(test_file.to_str().unwrap(), "New content");
        assert!(result.contains("written successfully"));
        let content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "New content");
        fs::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_edit_file_creates_parent_directories() {
        let temp_dir = env::temp_dir().join("test_edit_parent");
        let nested_path = temp_dir.join("nested/dir/file.txt");
        let _ = fs::remove_dir_all(&temp_dir);
        let result = edit_file(nested_path.to_str().unwrap(), "Nested content");
        assert!(result.contains("written successfully"));
        let content = fs::read_to_string(&nested_path).unwrap();
        assert_eq!(content, "Nested content");
        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_edit_file_empty_content() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_edit_empty.txt");
        let _ = fs::remove_file(&test_file);
        let result = edit_file(test_file.to_str().unwrap(), "");
        assert!(result.contains("written successfully"));
        let content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "");
        fs::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_edit_file_multiline_content() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_edit_multiline.txt");
        let content = "Line 1\nLine 2\nLine 3";
        let result = edit_file(test_file.to_str().unwrap(), content);
        assert!(result.contains("written successfully"));
        let read_content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(read_content, content);
        fs::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_edit_file_invalid_path() {
        let result = edit_file("/nonexistent/readonly/system/path/file.txt", "content");
        assert!(result.contains("Error"));
    }
}
