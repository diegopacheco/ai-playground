pub fn extract_specific_learnings(output: &str, task: &str) -> Vec<String> {
    let mut learnings = Vec::new();
    let output_lower = output.to_lowercase();
    if output_lower.contains("test") && (output_lower.contains("pass") || output_lower.contains("ok")) {
        learnings.push(format!("Tests passed for: {}", task.chars().take(50).collect::<String>()));
    }
    if output_lower.contains("build") && output_lower.contains("success") {
        learnings.push("Build succeeded without errors".to_string());
    }
    if output_lower.contains("lint") && !output_lower.contains("error") {
        learnings.push("Code passed linting checks".to_string());
    }
    if output_lower.contains("compiled") || output_lower.contains("bundled") {
        learnings.push("Compilation/bundling completed".to_string());
    }
    learnings
}

pub fn extract_mistakes_from_failure(error: &str, _output: &str) -> Vec<String> {
    let mut patterns = Vec::new();
    if error.contains("timeout") && !error.contains("likely") {
        patterns.push("Timeout - add progress indicators or async handling".to_string());
    }
    if error.contains("permission") {
        patterns.push("Permission denied - check file/directory permissions".to_string());
    }
    if error.contains("not found") || error.contains("No such file") {
        patterns.push("Missing dependency - verify paths and install deps".to_string());
    }
    if error.contains("syntax") || error.contains("parse") {
        patterns.push("Syntax error - validate code before execution".to_string());
    }
    if error.contains("memory") || error.contains("overflow") {
        patterns.push("Memory issue - avoid unbounded allocations".to_string());
    }
    if patterns.is_empty() && !error.is_empty() && !error.contains("likely") {
        let first_line = error.lines().next().unwrap_or("unknown error");
        if first_line.len() > 10 {
            patterns.push(format!("Error: {}", first_line.chars().take(100).collect::<String>()));
        }
    }
    patterns
}
