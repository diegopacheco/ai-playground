use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use chrono::Local;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;
use uuid::Uuid;

const LEARNING_CYCLES: u32 = 5;
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

struct CycleReport {
    cycle: u32,
    success: bool,
    learnings: Vec<String>,
    anti_patterns: Vec<String>,
    prompt_improved: bool,
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

fn add_learning(base_dir: &Path, learning: &str) -> String {
    let memory_path = base_dir.join(MEMORY_FILE);
    let entry = format!("- {}", learning);
    append_to_file(&memory_path, &entry);
    learning.to_string()
}

fn add_anti_pattern(base_dir: &Path, pattern: &str) -> String {
    let anti_path = base_dir.join(ANTI_PATTERN_FILE);
    let entry = format!("- {}", pattern);
    append_to_file(&anti_path, &entry);
    pattern.to_string()
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

async fn run_solution(project_dir: &Path) -> (bool, String) {
    let run_script = project_dir.join("run.sh");
    if !run_script.exists() {
        return (false, "No run.sh found in project directory".to_string());
    }
    println!("Running solution...");
    let output = Command::new("bash")
        .arg("run.sh")
        .current_dir(project_dir)
        .output()
        .await;
    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            if !stdout.is_empty() {
                println!("Output: {}", stdout);
            }
            if !stderr.is_empty() {
                eprintln!("Errors: {}", stderr);
            }
            if o.status.success() {
                (true, stdout)
            } else {
                (false, stderr)
            }
        }
        Err(e) => {
            (false, format!("Failed to run solution: {}", e))
        }
    }
}

fn extract_learnings_from_output(output: &str, solution_output: &str) -> Vec<String> {
    let mut learnings = Vec::new();
    if output.contains("successfully") || output.contains("completed") {
        learnings.push("Task completed successfully with current approach".to_string());
    }
    if output.contains("created") || output.contains("wrote") {
        learnings.push("File generation approach worked correctly".to_string());
    }
    if !solution_output.is_empty() && !solution_output.contains("error") {
        learnings.push("Generated code produced valid output".to_string());
    }
    if output.contains("test") && output.contains("pass") {
        learnings.push("Include tests in generated code for validation".to_string());
    }
    learnings
}

fn extract_anti_patterns_from_failure(error: &str, output: &str) -> Vec<String> {
    let mut patterns = Vec::new();
    if error.contains("timeout") {
        patterns.push("Avoid long-running operations without progress indicators".to_string());
    }
    if error.contains("permission") {
        patterns.push("Ensure proper file permissions before operations".to_string());
    }
    if error.contains("not found") {
        patterns.push("Verify dependencies and paths exist before use".to_string());
    }
    if error.contains("syntax") || error.contains("parse") {
        patterns.push("Validate syntax before execution".to_string());
    }
    if error.contains("memory") || error.contains("overflow") {
        patterns.push("Avoid unbounded data structures".to_string());
    }
    if output.contains("panic") || output.contains("crash") {
        patterns.push("Handle edge cases to prevent panics".to_string());
    }
    if error.contains("connection") || error.contains("network") {
        patterns.push("Add retry logic for network operations".to_string());
    }
    if patterns.is_empty() && !error.is_empty() {
        let first_line = error.lines().next().unwrap_or("unknown error");
        patterns.push(format!("Avoid: {}", first_line));
    }
    patterns
}

fn improve_prompt(base_dir: &Path, current_prompt: &str, error: &str, cycle: u32) -> String {
    let mut improved = current_prompt.to_string();
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    improved.push_str(&format!("\n\n## Improvement from cycle {} at {}:", cycle, timestamp));
    if error.contains("timeout") {
        improved.push_str("\n- Break tasks into smaller steps with progress output");
    }
    if error.contains("error") || error.contains("failed") {
        improved.push_str("\n- Add explicit error handling for all operations");
    }
    if error.contains("permission") {
        improved.push_str("\n- Check permissions before file operations");
    }
    if error.contains("not found") {
        improved.push_str("\n- Verify all dependencies exist before use");
    }
    if error.contains("syntax") {
        improved.push_str("\n- Double-check syntax before generating code");
    }
    if !error.is_empty() && !improved.contains("Improvement from cycle") {
        improved.push_str("\n- Be more careful with the approach");
    }
    archive_prompt(base_dir, current_prompt);
    update_current_prompt(base_dir, &improved);
    improved
}

fn print_cycle_report(report: &CycleReport) {
    println!("\n{}", "=".repeat(60));
    println!("CYCLE {} REPORT", report.cycle);
    println!("{}", "=".repeat(60));
    println!("Status: {}", if report.success { "SUCCESS" } else { "FAILED" });
    println!("\nLearnings acquired this cycle:");
    if report.learnings.is_empty() {
        println!("  (none)");
    } else {
        for learning in &report.learnings {
            println!("  + {}", learning);
        }
    }
    println!("\nAnti-patterns identified this cycle:");
    if report.anti_patterns.is_empty() {
        println!("  (none)");
    } else {
        for pattern in &report.anti_patterns {
            println!("  - {}", pattern);
        }
    }
    if report.prompt_improved {
        println!("\nPrompt was improved and archived for next cycle");
    }
    println!("{}", "=".repeat(60));
}

fn print_summary(reports: &[CycleReport]) {
    println!("\n{}", "#".repeat(60));
    println!("LEARNING SESSION SUMMARY");
    println!("{}", "#".repeat(60));
    let successes = reports.iter().filter(|r| r.success).count();
    let failures = reports.len() - successes;
    println!("Total cycles: {}", reports.len());
    println!("Successes: {}", successes);
    println!("Failures: {}", failures);
    let all_learnings: Vec<&String> = reports.iter().flat_map(|r| &r.learnings).collect();
    let all_anti_patterns: Vec<&String> = reports.iter().flat_map(|r| &r.anti_patterns).collect();
    println!("\nAll learnings accumulated:");
    if all_learnings.is_empty() {
        println!("  (none)");
    } else {
        for learning in &all_learnings {
            println!("  + {}", learning);
        }
    }
    println!("\nAll anti-patterns identified:");
    if all_anti_patterns.is_empty() {
        println!("  (none)");
    } else {
        for pattern in &all_anti_patterns {
            println!("  - {}", pattern);
        }
    }
    let improvements = reports.iter().filter(|r| r.prompt_improved).count();
    println!("\nPrompt versions created: {}", improvements);
    println!("{}", "#".repeat(60));
}

async fn run_learning_cycles(base_dir: &Path, task: &str, model: &str) -> Vec<CycleReport> {
    let mut reports = Vec::new();
    let project_name = sanitize_project_name(task);
    let base_project_dir = create_project_dir(base_dir, &project_name);
    println!("Base project directory: {}", base_project_dir.display());
    let mut current_prompt = read_current_prompt(base_dir);
    for cycle in 1..=LEARNING_CYCLES {
        println!("\n{}", "*".repeat(60));
        println!("LEARNING CYCLE {}/{}", cycle, LEARNING_CYCLES);
        println!("{}", "*".repeat(60));
        let cycle_dir = base_project_dir.join(format!("cycle-{}", cycle));
        let _ = fs::create_dir_all(&cycle_dir);
        let enhanced_prompt = build_enhanced_prompt(base_dir, task);
        let prompt_log = cycle_dir.join("prompt.txt");
        let _ = fs::write(&prompt_log, &enhanced_prompt);
        println!("\nExecuting agent for cycle {}...", cycle);
        let result = run_agent(&enhanced_prompt, &cycle_dir, model).await;
        let output_log = cycle_dir.join("output.txt");
        let _ = fs::write(&output_log, format!("STDOUT:\n{}\n\nSTDERR:\n{}", result.output, result.error));
        let mut report = CycleReport {
            cycle,
            success: result.success,
            learnings: Vec::new(),
            anti_patterns: Vec::new(),
            prompt_improved: false,
        };
        if result.success {
            println!("\nCycle {} agent completed successfully!", cycle);
            let (solution_success, solution_output) = run_solution(&cycle_dir).await;
            let learnings = extract_learnings_from_output(&result.output, &solution_output);
            for learning in &learnings {
                let saved = add_learning(base_dir, learning);
                report.learnings.push(saved);
            }
            if solution_success {
                let l = add_learning(base_dir, "Generated code executed without errors");
                report.learnings.push(l);
            } else {
                let patterns = extract_anti_patterns_from_failure(&solution_output, &result.output);
                for pattern in &patterns {
                    let saved = add_anti_pattern(base_dir, pattern);
                    report.anti_patterns.push(saved);
                }
                current_prompt = improve_prompt(base_dir, &current_prompt, &solution_output, cycle);
                report.prompt_improved = true;
            }
        } else {
            println!("\nCycle {} failed: {}", cycle, result.error);
            let patterns = extract_anti_patterns_from_failure(&result.error, &result.output);
            for pattern in &patterns {
                let saved = add_anti_pattern(base_dir, pattern);
                report.anti_patterns.push(saved);
            }
            current_prompt = improve_prompt(base_dir, &current_prompt, &result.error, cycle);
            report.prompt_improved = true;
        }
        print_cycle_report(&report);
        reports.push(report);
    }
    reports
}

fn print_help() {
    println!("Agent Learner - Self Learning Code Generation Agent");
    println!();
    println!("Usage: agent-learner [OPTIONS] [TASK]");
    println!();
    println!("Arguments:");
    println!("  [TASK]              The task description for code generation");
    println!("                      If not provided, enters REPL mode");
    println!();
    println!("Options:");
    println!("  --model <MODEL>     Claude model to use (default: sonnet)");
    println!("  --cycles <N>        Number of learning cycles (default: 5)");
    println!("  --repl              Enter interactive REPL mode");
    println!("  --list-prompts      Show all prompt versions");
    println!("  --show-memory       Show accumulated learnings");
    println!("  --show-anti-patterns Show anti-patterns to avoid");
    println!("  --help              Show this help message");
    println!();
    println!("REPL Commands:");
    println!("  :quit, :q           Exit the REPL");
    println!("  :memory, :m         Show current learnings");
    println!("  :anti, :a           Show current anti-patterns");
    println!("  :prompts, :p        Show prompt history");
    println!("  :help, :h           Show REPL help");
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

fn print_repl_help() {
    println!("REPL Commands:");
    println!("  :quit, :q           Exit the REPL");
    println!("  :memory, :m         Show current learnings");
    println!("  :anti, :a           Show current anti-patterns");
    println!("  :prompts, :p        Show prompt history");
    println!("  :clear              Clear screen");
    println!("  :help, :h           Show this help");
    println!();
    println!("Enter any other text to start a learning session with that task.");
}

async fn run_repl(base_dir: &Path, model: &str) {
    println!("Agent Learner REPL - Interactive Mode");
    println!("Type :help for commands, or enter a task to start learning");
    println!("Running {} learning cycles per task", LEARNING_CYCLES);
    println!();
    loop {
        print!("agent> ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            ":quit" | ":q" => {
                println!("Goodbye!");
                break;
            }
            ":memory" | ":m" => {
                show_memory(base_dir);
            }
            ":anti" | ":a" => {
                show_anti_patterns(base_dir);
            }
            ":prompts" | ":p" => {
                show_prompts(base_dir);
            }
            ":help" | ":h" => {
                print_repl_help();
            }
            ":clear" => {
                print!("\x1B[2J\x1B[1;1H");
                let _ = io::stdout().flush();
            }
            _ => {
                if input.starts_with(':') {
                    println!("Unknown command: {}", input);
                    println!("Type :help for available commands");
                } else {
                    println!("\nStarting learning session for: {}", input);
                    let reports = run_learning_cycles(base_dir, input, model).await;
                    print_summary(&reports);
                    println!();
                }
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let base_dir = get_base_dir();
    let mut model = "sonnet".to_string();
    let mut task: Option<String> = None;
    let mut repl_mode = false;
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
            "--repl" => {
                repl_mode = true;
            }
            "--model" => {
                if i + 1 < args.len() {
                    model = args[i + 1].clone();
                    i += 1;
                }
            }
            "--cycles" => {
                if i + 1 < args.len() {
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
    if repl_mode || task.is_none() {
        if args.len() < 2 {
            print_help();
            println!();
        }
        run_repl(&base_dir, &model).await;
        return;
    }
    let task = task.unwrap();
    println!("Agent Learner Starting...");
    println!("Task: {}", task);
    println!("Model: {}", model);
    println!("Learning cycles: {}", LEARNING_CYCLES);
    println!();
    let reports = run_learning_cycles(&base_dir, &task, &model).await;
    print_summary(&reports);
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
    fn test_extract_anti_patterns() {
        let patterns = extract_anti_patterns_from_failure("timeout occurred", "");
        assert!(!patterns.is_empty());
        let patterns = extract_anti_patterns_from_failure("permission denied", "");
        assert!(!patterns.is_empty());
        let patterns = extract_anti_patterns_from_failure("file not found", "");
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_extract_learnings() {
        let learnings = extract_learnings_from_output("Task completed successfully", "output");
        assert!(!learnings.is_empty());
    }
}
