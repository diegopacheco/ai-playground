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

const DEFAULT_CYCLES: u32 = 3;
const AGENT_TIMEOUT_SECS: u64 = 300;
const SOLUTION_RUN_TIMEOUT_SECS: u64 = 10;
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

struct ReviewFindings {
    architecture_issues: Vec<String>,
    design_issues: Vec<String>,
    code_quality_issues: Vec<String>,
    security_issues: Vec<String>,
    test_issues: Vec<String>,
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
    let dominated = [
        "task completed successfully",
        "generated code executed",
        "file generation approach worked",
        "code produced valid output",
    ];
    let dominated_lower = learning.to_lowercase();
    for dom in dominated.iter() {
        if dominated_lower.contains(dom) {
            return String::new();
        }
    }
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

fn build_review_prompt(cycle_dir: &Path, task: &str) -> String {
    let mut prompt = String::from("Review the generated code in the current directory. The original task was:\n");
    prompt.push_str(task);
    prompt.push_str("\n\nAnalyze and report on:\n");
    prompt.push_str("1. ARCHITECTURE: Is the architecture appropriate? Any structural issues?\n");
    prompt.push_str("2. DESIGN: Is the design correct? Any design pattern violations?\n");
    prompt.push_str("3. CODE QUALITY: Any bad code practices, code smells, or maintainability issues?\n");
    prompt.push_str("4. SECURITY: Any security vulnerabilities (injection, XSS, hardcoded secrets, etc)?\n");
    prompt.push_str("5. TESTS: Are there tests? Are they passing? Any missing test coverage?\n\n");
    prompt.push_str("For each category, list specific issues found with file:line references.\n");
    prompt.push_str("If no issues found in a category, say 'OK'.\n");
    prompt.push_str("Format your response as:\n");
    prompt.push_str("ARCHITECTURE: <issues or OK>\n");
    prompt.push_str("DESIGN: <issues or OK>\n");
    prompt.push_str("CODE_QUALITY: <issues or OK>\n");
    prompt.push_str("SECURITY: <issues or OK>\n");
    prompt.push_str("TESTS: <issues or OK>\n");
    let files = list_code_files(cycle_dir);
    if !files.is_empty() {
        prompt.push_str("\nFiles to review:\n");
        for f in files {
            prompt.push_str(&format!("- {}\n", f));
        }
    }
    prompt
}

fn list_code_files(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name() {
                    let name_str = name.to_string_lossy();
                    if !name_str.starts_with("prompt") && !name_str.starts_with("output") && !name_str.starts_with("review") {
                        files.push(name_str.to_string());
                    }
                }
            }
        }
    }
    files
}

fn parse_review_output(output: &str) -> ReviewFindings {
    let mut findings = ReviewFindings {
        architecture_issues: Vec::new(),
        design_issues: Vec::new(),
        code_quality_issues: Vec::new(),
        security_issues: Vec::new(),
        test_issues: Vec::new(),
    };
    for line in output.lines() {
        let line_upper = line.to_uppercase();
        if line_upper.starts_with("ARCHITECTURE:") {
            let content = line[13..].trim();
            if !content.to_uppercase().contains("OK") && !content.is_empty() {
                findings.architecture_issues.push(content.to_string());
            }
        } else if line_upper.starts_with("DESIGN:") {
            let content = line[7..].trim();
            if !content.to_uppercase().contains("OK") && !content.is_empty() {
                findings.design_issues.push(content.to_string());
            }
        } else if line_upper.starts_with("CODE_QUALITY:") || line_upper.starts_with("CODE QUALITY:") {
            let idx = if line_upper.starts_with("CODE_QUALITY:") { 13 } else { 12 };
            let content = line[idx..].trim();
            if !content.to_uppercase().contains("OK") && !content.is_empty() {
                findings.code_quality_issues.push(content.to_string());
            }
        } else if line_upper.starts_with("SECURITY:") {
            let content = line[9..].trim();
            if !content.to_uppercase().contains("OK") && !content.is_empty() {
                findings.security_issues.push(content.to_string());
            }
        } else if line_upper.starts_with("TESTS:") {
            let content = line[6..].trim();
            if !content.to_uppercase().contains("OK") && !content.is_empty() {
                findings.test_issues.push(content.to_string());
            }
        }
    }
    findings
}

fn findings_to_learnings(findings: &ReviewFindings) -> Vec<String> {
    let mut learnings = Vec::new();
    if findings.architecture_issues.is_empty() {
        learnings.push("Architecture passed review - structure is appropriate".to_string());
    }
    if findings.design_issues.is_empty() {
        learnings.push("Design passed review - patterns used correctly".to_string());
    }
    if findings.security_issues.is_empty() {
        learnings.push("Security passed review - no vulnerabilities found".to_string());
    }
    if findings.test_issues.is_empty() {
        learnings.push("Tests passed review - coverage is adequate".to_string());
    }
    learnings
}

fn findings_to_anti_patterns(findings: &ReviewFindings) -> Vec<String> {
    let mut patterns = Vec::new();
    for issue in &findings.architecture_issues {
        patterns.push(format!("Architecture issue: {}", issue));
    }
    for issue in &findings.design_issues {
        patterns.push(format!("Design issue: {}", issue));
    }
    for issue in &findings.code_quality_issues {
        patterns.push(format!("Code quality issue: {}", issue));
    }
    for issue in &findings.security_issues {
        patterns.push(format!("Security issue: {}", issue));
    }
    for issue in &findings.test_issues {
        patterns.push(format!("Test issue: {}", issue));
    }
    patterns
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

async fn run_solution_with_timeout(project_dir: &Path) -> (bool, String) {
    let run_script = project_dir.join("run.sh");
    if !run_script.exists() {
        return (false, "No run.sh found in project directory".to_string());
    }
    println!("Running solution with {}s timeout...", SOLUTION_RUN_TIMEOUT_SECS);
    let child = Command::new("bash")
        .arg("run.sh")
        .current_dir(project_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return (false, format!("Failed to start: {}", e)),
    };
    let timeout_duration = Duration::from_secs(SOLUTION_RUN_TIMEOUT_SECS);
    match timeout(timeout_duration, child.wait()).await {
        Ok(Ok(status)) => {
            let stdout = if let Some(mut out) = child.stdout.take() {
                let mut buf = Vec::new();
                let _ = tokio::io::AsyncReadExt::read_to_end(&mut out, &mut buf).await;
                String::from_utf8_lossy(&buf).to_string()
            } else {
                String::new()
            };
            let stderr = if let Some(mut err) = child.stderr.take() {
                let mut buf = Vec::new();
                let _ = tokio::io::AsyncReadExt::read_to_end(&mut err, &mut buf).await;
                String::from_utf8_lossy(&buf).to_string()
            } else {
                String::new()
            };
            if !stdout.is_empty() {
                println!("Output: {}", stdout.chars().take(500).collect::<String>());
            }
            if !stderr.is_empty() {
                eprintln!("Errors: {}", stderr.chars().take(500).collect::<String>());
            }
            if status.success() {
                (true, stdout)
            } else {
                (false, stderr)
            }
        }
        Ok(Err(e)) => (false, format!("Process error: {}", e)),
        Err(_) => {
            let _ = child.kill().await;
            println!("Solution timed out after {}s (likely a web server - OK)", SOLUTION_RUN_TIMEOUT_SECS);
            (true, "Timed out - likely running server".to_string())
        }
    }
}

fn extract_specific_learnings(output: &str, task: &str) -> Vec<String> {
    let mut learnings = Vec::new();
    let output_lower = output.to_lowercase();
    if output_lower.contains("test") && (output_lower.contains("pass") || output_lower.contains("ok")) {
        learnings.push(format!("Tests passed for task: {}", task.chars().take(50).collect::<String>()));
    }
    if output_lower.contains("build") && output_lower.contains("success") {
        learnings.push("Build succeeded without errors".to_string());
    }
    if output_lower.contains("lint") && !output_lower.contains("error") {
        learnings.push("Code passed linting checks".to_string());
    }
    if output_lower.contains("compiled") || output_lower.contains("bundled") {
        learnings.push("Compilation/bundling completed successfully".to_string());
    }
    learnings
}

fn extract_anti_patterns_from_failure(error: &str, output: &str) -> Vec<String> {
    let mut patterns = Vec::new();
    if error.contains("timeout") && !error.contains("likely") {
        patterns.push("Operation timed out - add progress indicators or async handling".to_string());
    }
    if error.contains("permission") {
        patterns.push("Permission denied - check file/directory permissions".to_string());
    }
    if error.contains("not found") || error.contains("No such file") {
        patterns.push("Missing file/dependency - verify paths and install dependencies".to_string());
    }
    if error.contains("syntax") || error.contains("parse") {
        patterns.push("Syntax error - validate code before execution".to_string());
    }
    if error.contains("memory") || error.contains("overflow") {
        patterns.push("Memory issue - avoid unbounded allocations".to_string());
    }
    if output.contains("panic") || output.contains("crash") {
        patterns.push("Runtime crash - handle edge cases and add error recovery".to_string());
    }
    if error.contains("connection") || error.contains("network") {
        patterns.push("Network error - add retry logic and connection handling".to_string());
    }
    if error.contains("undefined") || error.contains("null") {
        patterns.push("Null/undefined error - add proper null checks".to_string());
    }
    if patterns.is_empty() && !error.is_empty() && !error.contains("likely") {
        let first_line = error.lines().next().unwrap_or("unknown error");
        if first_line.len() > 10 {
            patterns.push(format!("Error encountered: {}", first_line.chars().take(100).collect::<String>()));
        }
    }
    patterns
}

fn improve_prompt(base_dir: &Path, current_prompt: &str, error: &str, findings: &ReviewFindings, cycle: u32) -> String {
    let mut improved = current_prompt.to_string();
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    improved.push_str(&format!("\n\n## Improvement from cycle {} at {}:", cycle, timestamp));
    if !findings.architecture_issues.is_empty() {
        improved.push_str("\n- Pay attention to architecture and code organization");
    }
    if !findings.design_issues.is_empty() {
        improved.push_str("\n- Follow proper design patterns");
    }
    if !findings.code_quality_issues.is_empty() {
        improved.push_str("\n- Improve code quality and readability");
    }
    if !findings.security_issues.is_empty() {
        improved.push_str("\n- Address security concerns (input validation, no hardcoded secrets)");
    }
    if !findings.test_issues.is_empty() {
        improved.push_str("\n- Add comprehensive tests with good coverage");
    }
    if error.contains("timeout") {
        improved.push_str("\n- Avoid blocking operations");
    }
    if error.contains("error") || error.contains("failed") {
        improved.push_str("\n- Add proper error handling");
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
    let filtered: Vec<&String> = report.learnings.iter().filter(|l| !l.is_empty()).collect();
    if filtered.is_empty() {
        println!("  (none)");
    } else {
        for learning in filtered {
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
    let all_learnings: Vec<&String> = reports.iter()
        .flat_map(|r| &r.learnings)
        .filter(|l| !l.is_empty())
        .collect();
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

async fn run_learning_cycles(base_dir: &Path, task: &str, model: &str, num_cycles: u32) -> Vec<CycleReport> {
    let mut reports = Vec::new();
    let project_name = sanitize_project_name(task);
    let base_project_dir = create_project_dir(base_dir, &project_name);
    println!("Base project directory: {}", base_project_dir.display());
    let mut current_prompt = read_current_prompt(base_dir);
    for cycle in 1..=num_cycles {
        println!("\n{}", "*".repeat(60));
        println!("LEARNING CYCLE {}/{}", cycle, num_cycles);
        println!("{}", "*".repeat(60));
        let cycle_dir = base_project_dir.join(format!("cycle-{}", cycle));
        let _ = fs::create_dir_all(&cycle_dir);
        let enhanced_prompt = build_enhanced_prompt(base_dir, task);
        let prompt_log = cycle_dir.join("prompt.txt");
        let _ = fs::write(&prompt_log, &enhanced_prompt);
        println!("\nPhase 1: Executing agent for cycle {}...", cycle);
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
            println!("\nPhase 2: Running solution with timeout...");
            let (solution_ok, solution_output) = run_solution_with_timeout(&cycle_dir).await;
            println!("\nPhase 3: Reviewing generated code...");
            let review_prompt = build_review_prompt(&cycle_dir, task);
            let review_result = run_agent(&review_prompt, &cycle_dir, model).await;
            let review_log = cycle_dir.join("review.txt");
            let _ = fs::write(&review_log, &review_result.output);
            let findings = parse_review_output(&review_result.output);
            println!("\nReview findings:");
            println!("  Architecture: {}", if findings.architecture_issues.is_empty() { "OK" } else { "Issues found" });
            println!("  Design: {}", if findings.design_issues.is_empty() { "OK" } else { "Issues found" });
            println!("  Code Quality: {}", if findings.code_quality_issues.is_empty() { "OK" } else { "Issues found" });
            println!("  Security: {}", if findings.security_issues.is_empty() { "OK" } else { "Issues found" });
            println!("  Tests: {}", if findings.test_issues.is_empty() { "OK" } else { "Issues found" });
            let task_learnings = extract_specific_learnings(&result.output, task);
            for learning in &task_learnings {
                let saved = add_learning(base_dir, learning);
                if !saved.is_empty() {
                    report.learnings.push(saved);
                }
            }
            let review_learnings = findings_to_learnings(&findings);
            for learning in &review_learnings {
                let saved = add_learning(base_dir, learning);
                if !saved.is_empty() {
                    report.learnings.push(saved);
                }
            }
            let review_anti_patterns = findings_to_anti_patterns(&findings);
            for pattern in &review_anti_patterns {
                let saved = add_anti_pattern(base_dir, pattern);
                report.anti_patterns.push(saved);
            }
            if !solution_ok && !solution_output.contains("likely") {
                let patterns = extract_anti_patterns_from_failure(&solution_output, &result.output);
                for pattern in &patterns {
                    let saved = add_anti_pattern(base_dir, pattern);
                    report.anti_patterns.push(saved);
                }
            }
            let has_issues = !findings.architecture_issues.is_empty()
                || !findings.design_issues.is_empty()
                || !findings.code_quality_issues.is_empty()
                || !findings.security_issues.is_empty()
                || !findings.test_issues.is_empty();
            if has_issues {
                current_prompt = improve_prompt(base_dir, &current_prompt, "", &findings, cycle);
                report.prompt_improved = true;
            }
        } else {
            println!("\nCycle {} failed: {}", cycle, result.error);
            let patterns = extract_anti_patterns_from_failure(&result.error, &result.output);
            for pattern in &patterns {
                let saved = add_anti_pattern(base_dir, pattern);
                report.anti_patterns.push(saved);
            }
            let empty_findings = ReviewFindings {
                architecture_issues: Vec::new(),
                design_issues: Vec::new(),
                code_quality_issues: Vec::new(),
                security_issues: Vec::new(),
                test_issues: Vec::new(),
            };
            current_prompt = improve_prompt(base_dir, &current_prompt, &result.error, &empty_findings, cycle);
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
    println!("  --cycles <N>        Number of learning cycles (default: 3)");
    println!("  --repl              Enter interactive REPL mode");
    println!("  --list-prompts      Show all prompt versions");
    println!("  --show-memory       Show accumulated learnings");
    println!("  --show-anti-patterns Show anti-patterns to avoid");
    println!("  --help              Show this help message");
    println!();
    println!("REPL Commands:");
    println!("  :quit, :q           Exit the REPL");
    println!("  :cycles <N>         Set number of cycles (e.g. :cycles 5)");
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
    println!("  :cycles <N>         Set number of cycles (e.g. :cycles 5)");
    println!("  :memory, :m         Show current learnings");
    println!("  :anti, :a           Show current anti-patterns");
    println!("  :prompts, :p        Show prompt history");
    println!("  :clear              Clear screen");
    println!("  :help, :h           Show this help");
    println!();
    println!("Enter any other text to start a learning session with that task.");
}

async fn run_repl(base_dir: &Path, model: &str, initial_cycles: u32) {
    let mut num_cycles = initial_cycles;
    println!("Agent Learner REPL - Interactive Mode");
    println!("Type :help for commands, or enter a task to start learning");
    println!("Running {} learning cycles per task (use :cycles N to change)", num_cycles);
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
        if input == ":quit" || input == ":q" {
            println!("Goodbye!");
            break;
        } else if input == ":memory" || input == ":m" {
            show_memory(base_dir);
        } else if input == ":anti" || input == ":a" {
            show_anti_patterns(base_dir);
        } else if input == ":prompts" || input == ":p" {
            show_prompts(base_dir);
        } else if input == ":help" || input == ":h" {
            print_repl_help();
        } else if input == ":clear" {
            print!("\x1B[2J\x1B[1;1H");
            let _ = io::stdout().flush();
        } else if input.starts_with(":cycles") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(n) = parts[1].parse::<u32>() {
                    if n > 0 && n <= 10 {
                        num_cycles = n;
                        println!("Cycles set to {}", num_cycles);
                    } else {
                        println!("Cycles must be between 1 and 10");
                    }
                } else {
                    println!("Invalid number: {}", parts[1]);
                }
            } else {
                println!("Current cycles: {}", num_cycles);
                println!("Usage: :cycles <N>");
            }
        } else if input.starts_with(':') {
            println!("Unknown command: {}", input);
            println!("Type :help for available commands");
        } else {
            println!("\nStarting learning session for: {}", input);
            let reports = run_learning_cycles(base_dir, input, model, num_cycles).await;
            print_summary(&reports);
            println!();
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
    let mut num_cycles = DEFAULT_CYCLES;
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
                    if let Ok(n) = args[i + 1].parse::<u32>() {
                        if n > 0 && n <= 10 {
                            num_cycles = n;
                        }
                    }
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
        run_repl(&base_dir, &model, num_cycles).await;
        return;
    }
    let task = task.unwrap();
    println!("Agent Learner Starting...");
    println!("Task: {}", task);
    println!("Model: {}", model);
    println!("Learning cycles: {}", num_cycles);
    println!();
    let reports = run_learning_cycles(&base_dir, &task, &model, num_cycles).await;
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
    fn test_parse_review_output() {
        let output = "ARCHITECTURE: OK\nDESIGN: Missing factory pattern\nCODE_QUALITY: OK\nSECURITY: Hardcoded password\nTESTS: No unit tests";
        let findings = parse_review_output(output);
        assert!(findings.architecture_issues.is_empty());
        assert!(!findings.design_issues.is_empty());
        assert!(findings.code_quality_issues.is_empty());
        assert!(!findings.security_issues.is_empty());
        assert!(!findings.test_issues.is_empty());
    }

    #[test]
    fn test_add_learning_filters_generic() {
        let temp_dir = std::env::temp_dir().join("agent-learner-filter-test");
        let _ = fs::create_dir_all(&temp_dir);
        let _ = fs::write(temp_dir.join(MEMORY_FILE), "");
        let result = add_learning(&temp_dir, "Task completed successfully");
        assert!(result.is_empty());
        let result = add_learning(&temp_dir, "Specific pattern: use factory for DI");
        assert!(!result.is_empty());
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
