use std::fs;
use std::path::Path;
use chrono::Local;
use crate::memory::read_file_or_default;

const PROMPTS_FILE: &str = "prompts.md";

pub const DEFAULT_PROMPT: &str = r#"# Current Prompt

You are a code generation agent. Your task is to generate working code based on user requirements.

Guidelines:
1. Generate complete, runnable code
2. Include a run.sh script to execute the code
3. Handle errors gracefully
4. Use best practices for the target language
5. Keep code simple and readable
6. Do not use external dependencies unless necessary

Output your code to the working directory provided.

# Past Prompts

"#;

pub fn ensure_prompts_exists(project_dir: &Path) {
    let prompt_path = project_dir.join(PROMPTS_FILE);
    if !prompt_path.exists() {
        let _ = fs::write(&prompt_path, DEFAULT_PROMPT);
    } else {
        let content = fs::read_to_string(&prompt_path).unwrap_or_default();
        if content.trim().is_empty() || !content.contains("# Current Prompt") {
            let _ = fs::write(&prompt_path, DEFAULT_PROMPT);
        }
    }
}

pub fn read_current_prompt(project_dir: &Path) -> String {
    ensure_prompts_exists(project_dir);
    let content = read_file_or_default(&project_dir.join(PROMPTS_FILE), DEFAULT_PROMPT);
    if let Some(start) = content.find("# Current Prompt") {
        if let Some(end) = content.find("# Past Prompts") {
            return content[start + 16..end].trim().to_string();
        }
    }
    content
}

pub fn archive_prompt(project_dir: &Path, old_prompt: &str) {
    ensure_prompts_exists(project_dir);
    let prompt_path = project_dir.join(PROMPTS_FILE);
    let content = read_file_or_default(&prompt_path, DEFAULT_PROMPT);
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let version = content.matches("## Version").count() + 1;
    let archived = format!("\n## Version {} - {}\n\n{}\n", version, timestamp, old_prompt);
    let new_content = content.replace("# Past Prompts", &format!("# Past Prompts{}", archived));
    let _ = fs::write(&prompt_path, new_content);
}

pub fn update_current_prompt(project_dir: &Path, new_prompt: &str) {
    ensure_prompts_exists(project_dir);
    let prompt_path = project_dir.join(PROMPTS_FILE);
    let content = read_file_or_default(&prompt_path, DEFAULT_PROMPT);
    if let Some(past_start) = content.find("# Past Prompts") {
        let past_section = &content[past_start..];
        let new_content = format!("# Current Prompt\n\n{}\n\n{}", new_prompt, past_section);
        let _ = fs::write(&prompt_path, new_content);
    }
}

pub fn build_enhanced_prompt(project_dir: &Path, user_task: &str) -> String {
    use crate::memory::{read_memory, read_mistakes};
    let current_prompt = read_current_prompt(project_dir);
    let memory = read_memory(project_dir);
    let mistakes = read_mistakes(project_dir);
    let mut enhanced = current_prompt.clone();
    if !memory.is_empty() {
        enhanced.push_str("\n\n## Learnings from past executions:\n");
        enhanced.push_str(&memory);
    }
    if !mistakes.is_empty() {
        enhanced.push_str("\n\n## Mistakes to avoid:\n");
        enhanced.push_str(&mistakes);
    }
    enhanced.push_str("\n\n## User Task:\n");
    enhanced.push_str(user_task);
    enhanced
}

pub fn show_prompts(project_dir: &Path) {
    ensure_prompts_exists(project_dir);
    let content = read_file_or_default(&project_dir.join(PROMPTS_FILE), DEFAULT_PROMPT);
    println!("{}", content);
}
