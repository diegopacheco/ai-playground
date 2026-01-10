use std::fs;

pub fn read_file(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => format!("Error reading file: {}", e),
    }
}
