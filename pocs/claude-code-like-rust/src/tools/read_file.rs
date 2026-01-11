use std::fs;

pub fn read_file(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => format!("Error reading file: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::env;

    #[test]
    fn test_read_file_success() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_read_file.txt");
        fs::write(&test_file, "Hello, World!").unwrap();
        let result = read_file(test_file.to_str().unwrap());
        assert_eq!(result, "Hello, World!");
        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_read_file_not_found() {
        let result = read_file("/nonexistent/path/to/file.txt");
        assert!(result.starts_with("Error reading file:"));
    }

    #[test]
    fn test_read_file_empty_path() {
        let result = read_file("");
        assert!(result.starts_with("Error reading file:"));
    }

    #[test]
    fn test_read_file_multiline_content() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_read_file_multiline.txt");
        let content = "Line 1\nLine 2\nLine 3";
        fs::write(&test_file, content).unwrap();
        let result = read_file(test_file.to_str().unwrap());
        assert_eq!(result, content);
        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_read_file_empty_content() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_read_file_empty.txt");
        fs::write(&test_file, "").unwrap();
        let result = read_file(test_file.to_str().unwrap());
        assert_eq!(result, "");
        fs::remove_file(test_file).unwrap();
    }
}
