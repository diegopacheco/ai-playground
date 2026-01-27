use std::collections::HashSet;
use std::fs;
use std::path::Path;

const MEMORY_FILE: &str = "memory.txt";
const MISTAKES_FILE: &str = "mistakes.txt";

pub fn read_file_or_default(path: &Path, default: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|_| default.to_string())
}

pub fn read_memory(project_dir: &Path) -> String {
    read_file_or_default(&project_dir.join(MEMORY_FILE), "")
}

pub fn read_mistakes(project_dir: &Path) -> String {
    read_file_or_default(&project_dir.join(MISTAKES_FILE), "")
}

fn get_existing_entries(path: &Path) -> HashSet<String> {
    let content = read_file_or_default(path, "");
    content.lines()
        .map(|l| l.trim_start_matches("- ").trim().to_lowercase())
        .filter(|l| !l.is_empty())
        .collect()
}

pub fn append_unique_to_file(path: &Path, content: &str) -> bool {
    let existing = get_existing_entries(path);
    let new_entry = content.trim_start_matches("- ").trim().to_lowercase();
    if existing.contains(&new_entry) {
        return false;
    }
    let file_content = read_file_or_default(path, "");
    let new_content = if file_content.is_empty() {
        content.to_string()
    } else {
        format!("{}\n{}", file_content.trim(), content)
    };
    let _ = fs::write(path, new_content);
    true
}

pub fn add_learning(project_dir: &Path, learning: &str) -> String {
    let dominated = [
        "task completed successfully",
        "generated code executed",
        "file generation approach worked",
        "code produced valid output",
        "passed review",
    ];
    let dominated_lower = learning.to_lowercase();
    for dom in dominated.iter() {
        if dominated_lower.contains(dom) {
            return String::new();
        }
    }
    let memory_path = project_dir.join(MEMORY_FILE);
    let entry = format!("- {}", learning);
    if append_unique_to_file(&memory_path, &entry) {
        learning.to_string()
    } else {
        String::new()
    }
}

pub fn add_mistake(project_dir: &Path, pattern: &str) -> String {
    let mistakes_path = project_dir.join(MISTAKES_FILE);
    let entry = format!("- {}", pattern);
    if append_unique_to_file(&mistakes_path, &entry) {
        pattern.to_string()
    } else {
        String::new()
    }
}

pub fn show_memory(project_dir: &Path) {
    let content = read_memory(project_dir);
    if content.is_empty() {
        println!("No learnings recorded yet");
    } else {
        println!("Accumulated Learnings:\n{}", content);
    }
}

pub fn show_mistakes(project_dir: &Path) {
    let content = read_mistakes(project_dir);
    if content.is_empty() {
        println!("No mistakes recorded yet");
    } else {
        println!("Mistakes to avoid:\n{}", content);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_entries() {
        let temp_dir = std::env::temp_dir().join("agent-learner-store-test");
        let _ = fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("test.txt");
        let _ = fs::write(&test_file, "");
        assert!(append_unique_to_file(&test_file, "- First entry"));
        assert!(!append_unique_to_file(&test_file, "- First entry"));
        assert!(append_unique_to_file(&test_file, "- Second entry"));
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
