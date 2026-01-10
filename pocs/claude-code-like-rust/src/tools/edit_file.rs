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
