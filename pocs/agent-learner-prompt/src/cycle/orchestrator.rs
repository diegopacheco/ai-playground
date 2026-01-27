use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::agents;
use crate::memory::{add_learning, add_mistake};
use crate::prompt::{ensure_prompts_exists, read_current_prompt, build_enhanced_prompt, archive_prompt, update_current_prompt};
use crate::review::{build_review_prompt, parse_review_output, findings_to_summary};
use crate::runner::run_solution_with_timeout;
use crate::learning::{build_learnings_prompt, build_mistakes_prompt, build_improve_prompt_prompt, parse_learnings, parse_mistakes};

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

fn copy_to_code_folder(project_dir: &Path, last_successful_cycle: u32) {
    let cycle_dir = project_dir.join(format!("cycle-{}", last_successful_cycle));
    let code_dir = project_dir.join(CODE_DIR);
    let _ = fs::create_dir_all(&code_dir);
    copy_dir_recursive(&cycle_dir, &code_dir);
    println!("Final code copied to: {}", code_dir.display());
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    let skip_files = ["prompt.txt", "output.txt", "review.txt", "learnings.txt", "mistakes.txt", "improved_prompt.txt"];
    if let Ok(entries) = fs::read_dir(src) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name() {
                let name_str = name.to_string_lossy();
                if skip_files.iter().any(|&s| name_str == s) {
                    continue;
                }
                let dest = dst.join(name);
                if path.is_dir() {
                    let _ = fs::create_dir_all(&dest);
                    copy_dir_recursive(&path, &dest);
                } else if path.is_file() {
                    let _ = fs::copy(&path, &dest);
                }
            }
        }
    }
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
            let solution_result = if solution_ok { 
                format!("Success: {}", solution_output.chars().take(500).collect::<String>())
            } else { 
                format!("Failed: {}", solution_output.chars().take(500).collect::<String>())
            };
            println!("\nPhase 3: Reviewing code...");
            let review_prompt = build_review_prompt(&cycle_dir, task);
            let review_result = agents::run_agent(agent, &review_prompt, model, &cycle_dir).await;
            let review_log = cycle_dir.join("review.txt");
            let _ = fs::write(&review_log, &review_result.output);
            let findings = parse_review_output(&review_result.output);
            let review_summary = findings_to_summary(&findings);
            println!("\nReview findings:");
            println!("  Architecture: {}", if findings.architecture_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.architecture_issues.len()) });
            println!("  Design: {}", if findings.design_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.design_issues.len()) });
            println!("  Code Quality: {}", if findings.code_quality_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.code_quality_issues.len()) });
            println!("  Security: {}", if findings.security_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.security_issues.len()) });
            println!("  Tests: {}", if findings.test_issues.is_empty() { "OK".to_string() } else { format!("{} issues", findings.test_issues.len()) });
            println!("\nPhase 4: Extracting learnings (LLM)...");
            let learnings_prompt = build_learnings_prompt(task, &result.output, &solution_result, &review_summary);
            let learnings_result = agents::run_agent(agent, &learnings_prompt, model, &cycle_dir).await;
            let learnings_log = cycle_dir.join("learnings.txt");
            let _ = fs::write(&learnings_log, &learnings_result.output);
            let llm_learnings = parse_learnings(&learnings_result.output);
            for learning in &llm_learnings {
                let saved = add_learning(&project_dir, learning);
                if !saved.is_empty() {
                    report.learnings.push(saved);
                }
            }
            println!("\nPhase 5: Extracting mistakes (LLM)...");
            let mistakes_prompt = build_mistakes_prompt(task, &result.output, &solution_result, &review_summary);
            let mistakes_result = agents::run_agent(agent, &mistakes_prompt, model, &cycle_dir).await;
            let mistakes_log = cycle_dir.join("mistakes.txt");
            let _ = fs::write(&mistakes_log, &mistakes_result.output);
            let llm_mistakes = parse_mistakes(&mistakes_result.output);
            for mistake in &llm_mistakes {
                let saved = add_mistake(&project_dir, mistake);
                if !saved.is_empty() {
                    report.mistakes.push(saved);
                }
            }
            println!("\nPhase 6: Improving prompt (LLM)...");
            let all_learnings = llm_learnings.join("\n");
            let all_mistakes = llm_mistakes.join("\n");
            let improve_prompt = build_improve_prompt_prompt(&current_prompt, &all_learnings, &all_mistakes, &review_summary);
            let improve_result = agents::run_agent(agent, &improve_prompt, model, &cycle_dir).await;
            let improve_log = cycle_dir.join("improved_prompt.txt");
            let _ = fs::write(&improve_log, &improve_result.output);
            if !improve_result.output.trim().is_empty() {
                archive_prompt(&project_dir, &current_prompt);
                update_current_prompt(&project_dir, &improve_result.output.trim());
                current_prompt = improve_result.output.trim().to_string();
                report.prompt_improved = true;
            }
        } else {
            println!("\nCycle {} failed: {}", cycle, result.error);
            let error_learning = format!("Cycle failed: {}", result.error.chars().take(200).collect::<String>());
            let saved = add_mistake(&project_dir, &error_learning);
            if !saved.is_empty() {
                report.mistakes.push(saved);
            }
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
