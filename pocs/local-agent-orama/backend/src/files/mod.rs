use crate::models::{FileEntry, FileContentResponse};
use std::path::Path;
use walkdir::WalkDir;
use tokio::fs;

pub fn list_files(worktree_path: &Path) -> Vec<FileEntry> {
    let mut files = Vec::new();
    if !worktree_path.exists() {
        return files;
    }
    for entry in WalkDir::new(worktree_path).min_depth(1).max_depth(10) {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.to_string_lossy().contains(".git") {
                continue;
            }
            let relative_path = path.strip_prefix(worktree_path).unwrap_or(path);
            let name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            files.push(FileEntry {
                path: relative_path.to_string_lossy().to_string(),
                name,
                is_dir: path.is_dir(),
            });
        }
    }
    files
}

pub async fn read_file(worktree_path: &Path, file_path: &str) -> Result<FileContentResponse, String> {
    let full_path = worktree_path.join(file_path);
    if !full_path.exists() {
        return Err("File not found".to_string());
    }
    if full_path.is_dir() {
        return Err("Path is a directory".to_string());
    }
    let content = fs::read_to_string(&full_path).await.map_err(|e| e.to_string())?;
    let language = detect_language(file_path);
    Ok(FileContentResponse { content, language })
}

fn detect_language(file_path: &str) -> String {
    let ext = Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "rs" => "rust",
        "js" => "javascript",
        "ts" => "typescript",
        "tsx" => "typescript",
        "jsx" => "javascript",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "c" => "c",
        "cpp" | "cc" | "cxx" => "cpp",
        "h" | "hpp" => "cpp",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "md" => "markdown",
        "html" => "html",
        "css" => "css",
        "sh" | "bash" => "bash",
        _ => "text",
    }.to_string()
}
