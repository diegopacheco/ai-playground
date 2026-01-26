use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use chrono::Local;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;
use uuid::Uuid;

const MAX_RETRIES: u32 = 3;
const AGENT_TIMEOUT_SECS: u64 = 300;
const MEMORY_FILE: &str = "memory.txt";
const ANTI_PATTERN_FILE: &str = "anti-pattern.txt";
const PROMPT_FILE: &str = "prompt.md";
const SOLUTIONS_DIR: &str = "solutions";

struct AgentResult {
    success: bool,
    output: String,
    error: String,
}

fn get_base_dir() -> PathBuf {
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn read_file_or_default(path: &Path, default: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|_| default.to_string())
}

fn read_memory(base_dir: &Path) -> String {
    read_file_or_default(&base_dir.join(MEMORY_FILE), "")
}

fn read_anti_patterns(base_dir: &Path) -> String {
    read_file_or_default(&base_dir.join(ANTI_PATTERN_FILE), "")
}

fn read_current_prompt(base_dir: &Path) -> String {
    let content = read_file_or_default(&base_dir.join(PROMPT_FILE), "");
    if let Some(start) = content.find("# Current Prompt") {
        if let Some(end) = content.find("# Past Prompts") {
            return content[start + 16..end].trim().to_string();
        }
    }
    content
}

fn append_to_file(path: &Path, content: &str) {
    let existing = read_file_or_default(path, "");
    let new_content = if existing.is_empty() {
        content.to_string()
    } else {
        format!("{}\n{}", existing.trim(), content)
    };
    let _ = fs::write(path, new_content);
}

fn archive_prompt(base_dir: &Path, old_prompt: &str) {
    let prompt_path = base_dir.join(PROMPT_FILE);
    let content = read_file_or_default(&prompt_path, "");
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let version = content.matches("## Version").count() + 1;
    let archived = format!("\n## Version {} - {}\n\n{}\n", version, timestamp, old_prompt);
    let new_content = content.replace("# Past Prompts", &format!("# Past Prompts{}", archived));
    let _ = fs::write(&prompt_path, new_content);
}

fn update_current_prompt(base_dir: &Path, new_prompt: &str) {
    let prompt_path = base_dir.join(PROMPT_FILE);
    let content = read_file_or_default(&prompt_path, "");
    if let Some(past_start) = content.find("# Past Prompts") {
        let past_section = &content[past_start..];
        let new_content = format!("# Current Prompt\n\n{}\n\n{}", new_prompt, past_section);
        let _ = fs::write(&prompt_path, new_content);
    }
}

fn add_learning(base_dir: &Path, learning: &str) {
    let memory_path = base_dir.join(MEMORY_FILE);
    append_to_file(&memory_path, &format!("- {}", learning));
}

fn add_anti_pattern(base_dir: &Path, pattern: &str) {
    let anti_path = base_dir.join(ANTI_PATTERN_FILE);
    append_to_file(&anti_path, &format!("- {}", pattern));
}

fn build_enhanced_prompt(base_dir: &Path, user_task: &str) -> String {
    let current_prompt = read_current_prompt(base_dir);
    let memory = read_memory(base_dir);
    let anti_patterns = read_anti_patterns(base_dir);
    let mut enhanced = current_prompt.clone();
    if !memory.is_empty() {
        enhanced.push_str("\n\n## Learnings from past executions:\n");
        enhanced.push_str(&memory);
    }
    if !anti_patterns.is_empty() {
        enhanced.push_str("\n\n## Anti-patterns to avoid:\n");
        enhanced.push_str(&anti_patterns);
    }
    enhanced.push_str("\n\n## User Task:\n");
    enhanced.push_str(user_task);
    enhanced
}

fn create_project_dir(base_dir: &Path, project_name: &str) -> PathBuf {
    let solutions_dir = base_dir.join(SOLUTIONS_DIR);
    let _ = fs::create_dir_all(&solutions_dir);
    let project_dir = solutions_dir.join(project_name);
    let _ = fs::create_dir_all(&project_dir);
    project_dir
}

fn sanitize_project_name(task: &str) -> String {
    let name: String = task
        .chars()
        .take(30)
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let name = name.trim_matches('-').to_string();
    if name.is_empty() {
        format!("project-{}", &Uuid::new_v4().to_string()[..8])
    } else {
        name
    }
}

async fn run_agent(prompt: &str, work_dir: &Path, model: &str) -> AgentResult {
    let mut child = match Command::new("claude")
        .arg("-p")
        .arg(prompt)
        .arg("--model")
        .arg(model)
        .arg("--dangerously-skip-permissions")
        .current_dir(work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return AgentResult {
                success: false,
                output: String::new(),
                error: format!("Failed to spawn agent: {}", e),
            };
        }
    };
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout_reader = BufReader::new(stdout);
    let stderr_reader = BufReader::new(stderr);
    let stdout_handle = tokio::spawn(async move {
        let mut lines = stdout_reader.lines();
        let mut output = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            println!("{}", line);
            output.push_str(&line);
            output.push('\n');
        }
        output
    });
    let stderr_handle = tokio::spawn(async move {
        let mut lines = stderr_reader.lines();
        let mut output = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            eprintln!("ERR: {}", line);
            output.push_str(&line);
            output.push('\n');
        }
        output
    });
    let timeout_duration = Duration::from_secs(AGENT_TIMEOUT_SECS);
    match timeout(timeout_duration, child.wait()).await {
        Ok(Ok(status)) => {
            let stdout_output = stdout_handle.await.unwrap_or_default();
            let stderr_output = stderr_handle.await.unwrap_or_default();
            AgentResult {
                success: status.success(),
                output: stdout_output,
                error: stderr_output,
            }
        }
        Ok(Err(e)) => AgentResult {
            success: false,
            output: String::new(),
            error: format!("Process error: {}", e),
        },
        Err(_) => {
            let _ = child.kill().await;
            AgentResult {
                success: false,
                output: String::new(),
                error: "Agent timed out".to_string(),
            }
        }
    }
}

async fn run_solution(project_dir: &Path) -> bool {
    let run_script = project_dir.join("run.sh");
    if !run_script.exists() {
        println!("No run.sh found in project directory");
        return false;
    }
    println!("Running solution...");
    let output = Command::new("bash")
        .arg("run.sh")
        .current_dir(project_dir)
        .output()
        .await;
    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            if !stdout.is_empty() {
                println!("Output: {}", stdout);
            }
            if !stderr.is_empty() {
                eprintln!("Errors: {}", stderr);
            }
            o.status.success()
        }
        Err(e) => {
            eprintln!("Failed to run solution: {}", e);
            false
        }
    }
}

fn extract_learning_from_success(output: &str) -> Option<String> {
    if output.contains("successfully") || output.contains("completed") {
        Some("Task completed successfully with current approach".to_string())
    } else {
        None
    }
}

fn extract_anti_pattern_from_failure(error: &str) -> Option<String> {
    if error.contains("timeout") {
        Some("Avoid long-running operations without progress indicators".to_string())
    } else if error.contains("permission") {
        Some("Ensure proper file permissions before operations".to_string())
    } else if error.contains("not found") {
        Some("Verify dependencies and paths exist before use".to_string())
    } else if !error.is_empty() {
        Some(format!("Avoid patterns that cause: {}", error.lines().next().unwrap_or("unknown error")))
    } else {
        None
    }
}

fn improve_prompt_after_failure(current_prompt: &str, error: &str) -> String {
    let mut improved = current_prompt.to_string();
    if error.contains("timeout") {
        improved.push_str("\n- Break tasks into smaller steps with progress output");
    }
    if error.contains("error") || error.contains("failed") {
        improved.push_str("\n- Add explicit error handling for all operations");
    }
    if error.contains("permission") {
        improved.push_str("\n- Check and request necessary permissions before file operations");
    }
    improved
}

fn print_help() {
    println!("Agent Learner - Self Learning Code Generation Agent");
    println!();
    println!("Usage: agent-learner [OPTIONS] <TASK>");
    println!();
    println!("Arguments:");
    println!("  <TASK>              The task description for code generation");
    println!();
    println!("Options:");
    println!("  --model <MODEL>     Claude model to use (default: sonnet)");
    println!("  --list-prompts      Show all prompt versions");
    println!("  --show-memory       Show accumulated learnings");
    println!("  --show-anti-patterns Show anti-patterns to avoid");
    println!("  --help              Show this help message");
}

fn show_prompts(base_dir: &Path) {
    let content = read_file_or_default(&base_dir.join(PROMPT_FILE), "No prompts found");
    println!("{}", content);
}

fn show_memory(base_dir: &Path) {
    let content = read_memory(base_dir);
    if content.is_empty() {
        println!("No learnings recorded yet");
    } else {
        println!("Accumulated Learnings:\n{}", content);
    }
}

fn show_anti_patterns(base_dir: &Path) {
    let content = read_anti_patterns(base_dir);
    if content.is_empty() {
        println!("No anti-patterns recorded yet");
    } else {
        println!("Anti-patterns to avoid:\n{}", content);
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let base_dir = get_base_dir();
    if args.len() < 2 {
        print_help();
        return;
    }
    let mut model = "sonnet".to_string();
    let mut task: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--list-prompts" => {
                show_prompts(&base_dir);
                return;
            }
            "--show-memory" => {
                show_memory(&base_dir);
                return;
            }
            "--show-anti-patterns" => {
                show_anti_patterns(&base_dir);
                return;
            }
            "--model" => {
                if i + 1 < args.len() {
                    model = args[i + 1].clone();
                    i += 1;
                }
            }
            _ => {
                if !args[i].starts_with('-') {
                    task = Some(args[i..].join(" "));
                    break;
                }
            }
        }
        i += 1;
    }
    let task = match task {
        Some(t) => t,
        None => {
            println!("Error: No task provided");
            print_help();
            return;
        }
    };
    println!("Agent Learner Starting...");
    println!("Task: {}", task);
    println!("Model: {}", model);
    println!();
    let project_name = sanitize_project_name(&task);
    let project_dir = create_project_dir(&base_dir, &project_name);
    println!("Project directory: {}", project_dir.display());
    let mut attempt = 0;
    let mut current_prompt = read_current_prompt(&base_dir);
    let mut last_error = String::new();
    while attempt < MAX_RETRIES {
        attempt += 1;
        println!("\n--- Attempt {}/{} ---", attempt, MAX_RETRIES);
        let enhanced_prompt = build_enhanced_prompt(&base_dir, &task);
        let prompt_log = project_dir.join(format!("prompt-attempt-{}.txt", attempt));
        let _ = fs::write(&prompt_log, &enhanced_prompt);
        let result = run_agent(&enhanced_prompt, &project_dir, &model).await;
        let output_log = project_dir.join(format!("output-attempt-{}.txt", attempt));
        let _ = fs::write(&output_log, format!("STDOUT:\n{}\n\nSTDERR:\n{}", result.output, result.error));
        if result.success {
            println!("\nAgent completed successfully!");
            if let Some(learning) = extract_learning_from_success(&result.output) {
                add_learning(&base_dir, &learning);
                println!("Added learning: {}", learning);
            }
            println!("\nRunning generated solution...");
            let solution_success = run_solution(&project_dir).await;
            if solution_success {
                println!("Solution executed successfully!");
                add_learning(&base_dir, "Generated code ran without errors");
            } else {
                println!("Solution execution failed");
                add_anti_pattern(&base_dir, "Generated code failed to execute");
            }
            return;
        }
        println!("Attempt {} failed: {}", attempt, result.error);
        last_error = result.error.clone();
        if let Some(anti_pattern) = extract_anti_pattern_from_failure(&result.error) {
            add_anti_pattern(&base_dir, &anti_pattern);
            println!("Added anti-pattern: {}", anti_pattern);
        }
        if attempt < MAX_RETRIES {
            current_prompt = improve_prompt_after_failure(&current_prompt, &result.error);
            println!("Improved prompt for next attempt");
        }
    }
    println!("\nAll {} attempts failed", MAX_RETRIES);
    println!("Last error: {}", last_error);
    archive_prompt(&base_dir, &current_prompt);
    let improved = improve_prompt_after_failure(&current_prompt, &last_error);
    update_current_prompt(&base_dir, &improved);
    println!("Archived old prompt and updated with improvements");
    println!("Project files saved to: {}", project_dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_project_name() {
        assert_eq!(sanitize_project_name("Hello World"), "hello-world");
        assert_eq!(sanitize_project_name("Test 123!@#"), "test-123");
        assert!(!sanitize_project_name("").is_empty());
    }

    #[test]
    fn test_build_enhanced_prompt() {
        let temp_dir = std::env::temp_dir().join("agent-learner-test");
        let _ = fs::create_dir_all(&temp_dir);
        let _ = fs::write(temp_dir.join(PROMPT_FILE), "# Current Prompt\n\nBase prompt\n\n# Past Prompts\n");
        let _ = fs::write(temp_dir.join(MEMORY_FILE), "- Learning 1");
        let _ = fs::write(temp_dir.join(ANTI_PATTERN_FILE), "- Anti 1");
        let enhanced = build_enhanced_prompt(&temp_dir, "Test task");
        assert!(enhanced.contains("Base prompt"));
        assert!(enhanced.contains("Learning 1"));
        assert!(enhanced.contains("Anti 1"));
        assert!(enhanced.contains("Test task"));
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_extract_anti_pattern() {
        assert!(extract_anti_pattern_from_failure("timeout occurred").is_some());
        assert!(extract_anti_pattern_from_failure("permission denied").is_some());
        assert!(extract_anti_pattern_from_failure("file not found").is_some());
        assert!(extract_anti_pattern_from_failure("").is_none());
    }
}
