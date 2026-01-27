use std::fs;
use std::path::{Path, PathBuf};
use chrono::Local;
use uuid::Uuid;

use crate::agents;
use crate::memory::{add_learning, add_mistake};
use crate::prompt::{ensure_prompts_exists, read_current_prompt, build_enhanced_prompt, archive_prompt, update_current_prompt};
use crate::review::{ReviewFindings, build_review_prompt, parse_review_output, findings_to_mistakes};
use crate::runner::run_solution_with_timeout;
use crate::learning::{extract_specific_learnings, extract_mistakes_from_failure};

const SOLUTIONS_DIR: &str = "solutions";
const CODE_DIR: &str = "code";

pub struct CycleReport {
    pub cycle: u32,
    pub success: bool,
    pub learnings: Vec<String>,
    pub mistakes: Vec<String>,
    pub prompt_improved: bool,
}

pub fn sanitize_project_name(task: &str) -> String {
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

pub fn create_project_dir(base_dir: &Path, project_name: &str) -> PathBuf {
    let solutions_dir = base_dir.join(SOLUTIONS_DIR);
    let _ = fs::create_dir_all(&solutions_dir);
    let project_dir = solutions_dir.join(project_name);
    let _ = fs::create_dir_all(&project_dir);
    project_dir
}

fn improve_prompt(project_dir: &Path, current_prompt: &str, error: &str, findings: &ReviewFindings, cycle: u32) -> String {
    let mut improved = current_prompt.to_string();
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    improved.push_str(&format!("\n\n## Improvements from cycle {} at {}:", cycle, timestamp));
    let mut added = false;
    if !findings.architecture_issues.is_empty() {
        improved.push_str("\n- Pay attention to architecture and code organization");
        added = true;
    }
    if !findings.design_issues.is_empty() {
        improved.push_str("\n- Follow proper design patterns");
        added = true;
    }
    if !findings.code_quality_issues.is_empty() {
        improved.push_str("\n- Improve code quality and readability");
        added = true;
    }
    if !findings.security_issues.is_empty() {
        improved.push_str("\n- Address security concerns (input validation, no hardcoded secrets)");
        added = true;
    }
    if !findings.test_issues.is_empty() {
        improved.push_str("\n- Add comprehensive tests with good coverage");
        added = true;
    }
    if error.contains("timeout") {
        improved.push_str("\n- Avoid blocking operations");
        added = true;
    }
    if error.contains("error") || error.contains("failed") {
        improved.push_str("\n- Add proper error handling");
        added = true;
    }
    if !added {
        improved.push_str("\n- Continue with current approach");
    }
    archive_prompt(project_dir, current_prompt);
    update_current_prompt(project_dir, &improved);
    improved
}

fn copy_to_code_folder(project_dir: &Path, last_successful_cycle: u32) {
    let cycle_dir = project_dir.join(format!("cycle-{}", last_successful_cycle));
    let code_dir = project_dir.join(CODE_DIR);
    let _ = fs::create_dir_all(&code_dir);
    if let Ok(entries) = fs::read_dir(&cycle_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name() {
                    let name_str = name.to_string_lossy();
                    if !name_str.starts_with("prompt") && !name_str.starts_with("output") && !name_str.starts_with("review") {
                        let dest = code_dir.join(name);
                        let _ = fs::copy(&path, &dest);
                    }
                }
            }
        }
    }
    println!("Final code copied to: {}", code_dir.display());
}

pub fn print_cycle_report(report: &CycleReport) {
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
    println!("\nMistakes identified this cycle:");
    let mistakes_filtered: Vec<&String> = report.mistakes.iter().filter(|l| !l.is_empty()).collect();
    if mistakes_filtered.is_empty() {
        println!("  (none)");
    } else {
        for pattern in mistakes_filtered {
            println!("  - {}", pattern);
        }
    }
    if report.prompt_improved {
        println!("\nPrompt was improved and archived for next cycle");
    }
    println!("{}", "=".repeat(60));
}

pub fn print_summary(reports: &[CycleReport]) {
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
    let all_mistakes: Vec<&String> = reports.iter()
        .flat_map(|r| &r.mistakes)
        .filter(|l| !l.is_empty())
        .collect();
    println!("\nLearnings accumulated:");
    if all_learnings.is_empty() {
        println!("  (none)");
    } else {
        for learning in &all_learnings {
            println!("  + {}", learning);
        }
    }
    println!("\nMistakes identified:");
    if all_mistakes.is_empty() {
        println!("  (none)");
    } else {
        for pattern in &all_mistakes {
            println!("  - {}", pattern);
        }
    }
    let improvements = reports.iter().filter(|r| r.prompt_improved).count();
    println!("\nPrompt versions created: {}", improvements);
    println!("{}", "#".repeat(60));
}

pub fn get_solutions_dir() -> &'static str {
    SOLUTIONS_DIR
}

pub async fn run_learning_cycles(base_dir: &Path, task: &str, agent: &str, model: &str, num_cycles: u32) -> Vec<CycleReport> {
    let mut reports = Vec::new();
    let project_name = sanitize_project_name(task);
    let project_dir = create_project_dir(base_dir, &project_name);
    println!("Project directory: {}", project_dir.display());
    ensure_prompts_exists(&project_dir);
    let mut current_prompt = read_current_prompt(&project_dir);
    let mut last_successful_cycle = 0u32;
    for cycle in 1..=num_cycles {
        println!("\n{}", "*".repeat(60));
        println!("LEARNING CYCLE {}/{} [Agent: {} | Model: {}]", cycle, num_cycles, agent, model);
        println!("{}", "*".repeat(60));
        let cycle_dir = project_dir.join(format!("cycle-{}", cycle));
        let _ = fs::create_dir_all(&cycle_dir);
        let enhanced_prompt = build_enhanced_prompt(&project_dir, task);
        let prompt_log = cycle_dir.join("prompt.txt");
        let _ = fs::write(&prompt_log, &enhanced_prompt);
        println!("\nPhase 1: Generating code...");
        let result = agents::run_agent(agent, &enhanced_prompt, model, &cycle_dir).await;
        let output_log = cycle_dir.join("output.txt");
        let _ = fs::write(&output_log, format!("STDOUT:\n{}\n\nSTDERR:\n{}", result.output, result.error));
        let mut report = CycleReport {
            cycle,
            success: result.success,
            learnings: Vec::new(),
            mistakes: Vec::new(),
            prompt_improved: false,
        };
        if result.success {
            last_successful_cycle = cycle;
            println!("\nPhase 2: Running solution...");
            let (solution_ok, solution_output) = run_solution_with_timeout(&cycle_dir).await;
            println!("\nPhase 3: Reviewing code...");
            let review_prompt = build_review_prompt(&cycle_dir, task);
            let review_result = agents::run_agent(agent, &review_prompt, model, &cycle_dir).await;
            let review_log = cycle_dir.join("review.txt");
            let _ = fs::write(&review_log, &review_result.output);
            let findings = parse_review_output(&review_result.output);
            println!("\nReview findings:");
            println!("  Architecture: {}", if findings.architecture_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.architecture_issues.len()) });
            println!("  Design: {}", if findings.design_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.design_issues.len()) });
            println!("  Code Quality: {}", if findings.code_quality_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.code_quality_issues.len()) });
            println!("  Security: {}", if findings.security_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.security_issues.len()) });
            println!("  Tests: {}", if findings.test_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.test_issues.len()) });
            let task_learnings = extract_specific_learnings(&result.output, task);
            for learning in &task_learnings {
                let saved = add_learning(&project_dir, learning);
                if !saved.is_empty() {
                    report.learnings.push(saved);
                }
            }
            let review_mistakes = findings_to_mistakes(&findings);
            for pattern in &review_mistakes {
                let saved = add_mistake(&project_dir, pattern);
                if !saved.is_empty() {
                    report.mistakes.push(saved);
                }
            }
            if !solution_ok && !solution_output.contains("likely") {
                let patterns = extract_mistakes_from_failure(&solution_output, &result.output);
                for pattern in &patterns {
                    let saved = add_mistake(&project_dir, pattern);
                    if !saved.is_empty() {
                        report.mistakes.push(saved);
                    }
                }
            }
            current_prompt = improve_prompt(&project_dir, &current_prompt, "", &findings, cycle);
            report.prompt_improved = true;
        } else {
            println!("\nCycle {} failed: {}", cycle, result.error);
            let patterns = extract_mistakes_from_failure(&result.error, &result.output);
            for pattern in &patterns {
                let saved = add_mistake(&project_dir, pattern);
                if !saved.is_empty() {
                    report.mistakes.push(saved);
                }
            }
            let empty_findings = ReviewFindings::default();
            current_prompt = improve_prompt(&project_dir, &current_prompt, &result.error, &empty_findings, cycle);
            report.prompt_improved = true;
        }
        print_cycle_report(&report);
        reports.push(report);
    }
    if last_successful_cycle > 0 {
        copy_to_code_folder(&project_dir, last_successful_cycle);
    }
    reports
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
}
