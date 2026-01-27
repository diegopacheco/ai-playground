use std::fs;
use std::path::Path;

#[derive(Default)]
pub struct ReviewFindings {
    pub architecture_issues: Vec<String>,
    pub design_issues: Vec<String>,
    pub code_quality_issues: Vec<String>,
    pub security_issues: Vec<String>,
    pub test_issues: Vec<String>,
}

pub fn build_review_prompt(cycle_dir: &Path, task: &str) -> String {
    let mut prompt = String::from("Review the generated code in the current directory. The original task was:\n");
    prompt.push_str(task);
    prompt.push_str("\n\nAnalyze and report on:\n");
    prompt.push_str("1. ARCHITECTURE: Is the architecture appropriate? Any structural issues?\n");
    prompt.push_str("2. DESIGN: Is the design correct? Any design pattern violations?\n");
    prompt.push_str("3. CODE QUALITY: Any bad code practices, code smells, or maintainability issues?\n");
    prompt.push_str("4. SECURITY: Any security vulnerabilities (injection, XSS, hardcoded secrets, etc)?\n");
    prompt.push_str("5. TESTS: Are there tests? Are they passing? Any missing test coverage?\n\n");
    prompt.push_str("For each category, list specific issues found.\n");
    prompt.push_str("IMPORTANT: Use this EXACT format (no markdown, no bold):\n");
    prompt.push_str("ARCHITECTURE: OK or ARCHITECTURE: <issue description>\n");
    prompt.push_str("DESIGN: OK or DESIGN: <issue description>\n");
    prompt.push_str("CODE_QUALITY: OK or CODE_QUALITY: <issue description>\n");
    prompt.push_str("SECURITY: OK or SECURITY: <issue description>\n");
    prompt.push_str("TESTS: OK or TESTS: <issue description>\n");
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

fn normalize_category(line: &str) -> Option<(&str, String)> {
    let line_clean = line.replace("**", "").replace("*", "");
    let line_upper = line_clean.to_uppercase();
    let categories = [
        ("ARCHITECTURE", "architecture"),
        ("DESIGN", "design"),
        ("CODE_QUALITY", "code_quality"),
        ("CODE QUALITY", "code_quality"),
        ("SECURITY", "security"),
        ("TESTS", "tests"),
        ("TEST", "tests"),
    ];
    for (prefix, cat) in categories.iter() {
        if line_upper.contains(prefix) && line_clean.contains(':') {
            if let Some(idx) = line_clean.find(':') {
                let content = line_clean[idx + 1..].trim().to_string();
                return Some((cat, content));
            }
        }
    }
    None
}

pub fn has_issues(content: &str) -> bool {
    let content_lower = content.to_lowercase();
    if content_lower == "ok" || content_lower.is_empty() {
        return false;
    }
    let issue_indicators = [
        "issue", "problem", "error", "missing", "no test", "not found",
        "vulnerability", "hardcoded", "deprecated", "warning", "critical",
        "unsafe", "invalid", "incorrect", "fail", "bug", "flaw",
    ];
    for indicator in issue_indicators.iter() {
        if content_lower.contains(indicator) {
            return true;
        }
    }
    content_lower != "ok" && content.len() > 10
}

pub fn parse_review_output(output: &str) -> ReviewFindings {
    let mut findings = ReviewFindings::default();
    let mut current_category: Option<&str> = None;
    for line in output.lines() {
        let line_trimmed = line.trim();
        if line_trimmed.is_empty() {
            continue;
        }
        if let Some((cat, content)) = normalize_category(line_trimmed) {
            current_category = Some(cat);
            if has_issues(&content) {
                match cat {
                    "architecture" => findings.architecture_issues.push(content),
                    "design" => findings.design_issues.push(content),
                    "code_quality" => findings.code_quality_issues.push(content),
                    "security" => findings.security_issues.push(content),
                    "tests" => findings.test_issues.push(content),
                    _ => {}
                }
            }
        } else if let Some(cat) = current_category {
            if line_trimmed.starts_with('-') || line_trimmed.starts_with('*') {
                let content = line_trimmed.trim_start_matches(|c| c == '-' || c == '*' || c == ' ').to_string();
                if has_issues(&content) && content.len() > 5 {
                    match cat {
                        "architecture" => findings.architecture_issues.push(content),
                        "design" => findings.design_issues.push(content),
                        "code_quality" => findings.code_quality_issues.push(content),
                        "security" => findings.security_issues.push(content),
                        "tests" => findings.test_issues.push(content),
                        _ => {}
                    }
                }
            }
        }
    }
    findings
}

pub fn findings_to_summary(findings: &ReviewFindings) -> String {
    let mut parts = Vec::new();
    if !findings.architecture_issues.is_empty() {
        parts.push(format!("Architecture: {}", findings.architecture_issues.join("; ")));
    }
    if !findings.design_issues.is_empty() {
        parts.push(format!("Design: {}", findings.design_issues.join("; ")));
    }
    if !findings.code_quality_issues.is_empty() {
        parts.push(format!("Code Quality: {}", findings.code_quality_issues.join("; ")));
    }
    if !findings.security_issues.is_empty() {
        parts.push(format!("Security: {}", findings.security_issues.join("; ")));
    }
    if !findings.test_issues.is_empty() {
        parts.push(format!("Tests: {}", findings.test_issues.join("; ")));
    }
    if parts.is_empty() {
        "All checks passed - no issues found".to_string()
    } else {
        parts.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_review_with_markdown() {
        let output = "**ARCHITECTURE**: Multiple issues found\n- backend/main.go:36: Global state\n**SECURITY**: Critical issues\n- Hardcoded password";
        let findings = parse_review_output(output);
        assert!(!findings.architecture_issues.is_empty());
        assert!(!findings.security_issues.is_empty());
    }

    #[test]
    fn test_parse_review_plain() {
        let output = "ARCHITECTURE: OK\nDESIGN: Missing pattern\nSECURITY: Hardcoded secret\nTESTS: No tests";
        let findings = parse_review_output(output);
        assert!(findings.architecture_issues.is_empty());
        assert!(!findings.design_issues.is_empty());
        assert!(!findings.security_issues.is_empty());
        assert!(!findings.test_issues.is_empty());
    }

    #[test]
    fn test_has_issues() {
        assert!(!has_issues("OK"));
        assert!(!has_issues("ok"));
        assert!(has_issues("Missing error handling"));
        assert!(has_issues("No tests found"));
        assert!(has_issues("Hardcoded password vulnerability"));
    }
}
