pub fn build_learnings_prompt(task: &str, output: &str, solution_result: &str, review_summary: &str) -> String {
    format!(r#"Analyze the code generation cycle that just completed. The task was: {}

Review the generated code and execution results:
- Code output: {}
- Solution run result: {}
- Review findings: {}

What specific, actionable learnings can be extracted from this cycle?
Focus on patterns that worked well and should be repeated.
Do NOT include generic statements like "task completed successfully".

Output format (one learning per line, be specific):
LEARNING: <specific actionable insight>
LEARNING: <specific actionable insight>"#, task, output.chars().take(2000).collect::<String>(), solution_result, review_summary)
}

pub fn build_mistakes_prompt(task: &str, output: &str, solution_result: &str, review_summary: &str) -> String {
    format!(r#"Analyze the code generation cycle for mistakes to avoid. The task was: {}

Review the generated code and execution results:
- Code output: {}
- Solution run result: {}
- Review findings: {}

What specific mistakes were made that should be avoided in future cycles?
Focus on concrete anti-patterns, not generic advice.

Output format (one mistake per line, be specific):
MISTAKE: <specific mistake to avoid>
MISTAKE: <specific mistake to avoid>"#, task, output.chars().take(2000).collect::<String>(), solution_result, review_summary)
}

pub fn build_improve_prompt_prompt(current_prompt: &str, learnings: &str, mistakes: &str, review_summary: &str) -> String {
    format!(r#"You are a prompt engineer. Your task is to improve the code generation prompt.

Current prompt:
{}

Learnings from this cycle:
{}

Mistakes identified:
{}

Review findings:
{}

Generate an improved version of the prompt that:
1. Incorporates the learnings as guidelines
2. Explicitly warns against the identified mistakes
3. Addresses the review findings
4. Remains clear and actionable

Output ONLY the improved prompt text, no explanations."#, current_prompt, learnings, mistakes, review_summary)
}

pub fn parse_learnings(output: &str) -> Vec<String> {
    output.lines()
        .filter(|line| line.trim().starts_with("LEARNING:"))
        .map(|line| line.trim().trim_start_matches("LEARNING:").trim().to_string())
        .filter(|s| !s.is_empty() && s.len() > 10)
        .collect()
}

pub fn parse_mistakes(output: &str) -> Vec<String> {
    output.lines()
        .filter(|line| line.trim().starts_with("MISTAKE:"))
        .map(|line| line.trim().trim_start_matches("MISTAKE:").trim().to_string())
        .filter(|s| !s.is_empty() && s.len() > 10)
        .collect()
}

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
