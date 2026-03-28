use crate::agents;
use crate::detect::{detect_project, test_command};
use crate::pr;
use crate::state::{ActionType, AgentAction, AgentLog, SharedState, now_timestamp};
use std::process::Command;

pub fn check_and_fix_tests(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64, state: &SharedState, dry_run: bool,
) -> Result<(), String> {
    let detected = detect_project(clone_path);
    let project_root = &detected.project_root;
    let (cmd, args) = test_command(&detected.project_type);

    let output = Command::new(cmd)
        .args(&args)
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("Test command failed: {}", e))?;

    if output.status.success() {
        check_test_gaps(clone_path, agent, model, owner, repo, pr_number, state, dry_run)?;
        return Ok(());
    }

    let mut test_error = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    for attempt in 0..10 {
        let source_files = pr::list_source_files(project_root);
        let mut file_context = String::new();
        for f in &source_files {
            if let Ok(content) = pr::read_file(f) {
                file_context.push_str(&format!("File: {}\n{}\n\n", f, content));
            }
        }

        let prompt = format!(
            "You are a senior developer. The following tests are failing.\n\
            Fix the code or tests and return ONLY the corrected file content inside a code block.\n\
            Include the file path as the first line of each code block like: // FILE: path/to/file\n\n\
            Project type: {}\nTest output:\n{}\n\nSource files:\n{}",
            detected.project_type, test_error, file_context
        );

        let response = agents::run_llm(agent, model, &prompt)?;

        if let Some(code) = agents::extract_code_block(&response) {
            let file_path = extract_file_path(&code, &source_files, project_root);
            if let Some(target_file) = file_path {
                let clean_code = remove_file_path_line(&code);
                pr::write_file(&target_file, &clean_code)?;

                let mut st = state.lock().unwrap();
                let id = st.next_id();
                st.add_action(AgentAction {
                    id,
                    timestamp: now_timestamp(),
                    action_type: ActionType::TestFix,
                    description: format!("Test fix attempt {}", attempt + 1),
                    files_changed: vec![target_file.clone()],
                    llm_agent: agent.to_string(),
                    llm_model: model.to_string(),
                    commit_sha: None,
                });
                st.add_log(AgentLog {
                    id,
                    timestamp: now_timestamp(),
                    action_type: ActionType::TestFix,
                    llm_agent: agent.to_string(),
                    llm_model: model.to_string(),
                    prompt: prompt.clone(),
                    response: response.clone(),
                    result: format!("Applied test fix to {}", target_file),
                    commit_sha: None,
                });
                st.counters.test_fixes += 1;
                let last_action = st.actions.last().unwrap().clone();
                let counters = st.counters.clone();
                st.broadcast_sse_json("action", &last_action);
                st.broadcast_sse_json("counter_update", &counters);
                drop(st);
            }
        }

        let retry = Command::new(cmd)
            .args(&args)
            .current_dir(project_root)
            .output()
            .map_err(|e| format!("Test retry failed: {}", e))?;

        if retry.status.success() {
            if !dry_run {
                pr::git_add_commit_push(clone_path, "[agent-pr] fix: test failures")?;
            } else {
                println!("[dry-run] Skipping commit/push for test fix");
            }
            return Ok(());
        }

        test_error = format!(
            "{}{}",
            String::from_utf8_lossy(&retry.stdout),
            String::from_utf8_lossy(&retry.stderr)
        );
    }

    Err("Test fix failed after 10 attempts".to_string())
}

fn check_test_gaps(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64, state: &SharedState, dry_run: bool,
) -> Result<(), String> {
    let detected = detect_project(clone_path);
    let project_root = &detected.project_root;
    let pr_diff = pr::get_pr_diff(owner, repo, pr_number).unwrap_or_default();
    let source_files = pr::list_source_files(project_root);
    let mut file_context = String::new();
    let mut test_files = String::new();

    for f in &source_files {
        if let Ok(content) = pr::read_file(f) {
            if f.contains("test") || f.contains("spec") {
                test_files.push_str(&format!("File: {}\n{}\n\n", f, content));
            } else {
                file_context.push_str(&format!("File: {}\n{}\n\n", f, content));
            }
        }
    }

    let prompt = format!(
        "You are a senior developer. Review the following PR diff and source code.\n\
        Identify if there are missing tests for the changed code.\n\
        If there are test gaps, write the test file content inside a code block.\n\
        Include the file path as the first line like: // FILE: path/to/test_file\n\
        Also on the VERY FIRST line of your response (before any code block), write a classification line like:\n\
        TEST_TYPE: unit=3,integration=1,e2e=0,other=0\n\
        Where each number is how many tests of that type you are adding.\n\
        If no gaps, respond with \"NO_GAPS\".\n\n\
        Project type: {}\nPR diff:\n{}\n\nSource files:\n{}\n\nExisting test files:\n{}",
        detected.project_type, pr_diff, file_context, test_files
    );

    let response = agents::run_llm(agent, model, &prompt)?;

    if response.contains("NO_GAPS") {
        return Ok(());
    }

    let classification = parse_test_classification(&response);

    if let Some(code) = agents::extract_code_block(&response) {
        let file_path = extract_file_path(&code, &source_files, project_root);
        if let Some(target_file) = file_path {
            let clean_code = remove_file_path_line(&code);
            pr::write_file(&target_file, &clean_code)?;

            let desc = format!(
                "Added missing tests (unit: {}, integration: {}, e2e: {}, other: {})",
                classification.0, classification.1, classification.2, classification.3
            );

            let mut st = state.lock().unwrap();
            let id = st.next_id();
            st.add_action(AgentAction {
                id,
                timestamp: now_timestamp(),
                action_type: ActionType::TestAdd,
                description: desc,
                files_changed: vec![target_file.clone()],
                llm_agent: agent.to_string(),
                llm_model: model.to_string(),
                commit_sha: None,
            });
            st.add_log(AgentLog {
                id,
                timestamp: now_timestamp(),
                action_type: ActionType::TestAdd,
                llm_agent: agent.to_string(),
                llm_model: model.to_string(),
                prompt: prompt.clone(),
                response: response.clone(),
                result: format!(
                    "Added tests to {} (unit: {}, integration: {}, e2e: {}, other: {})",
                    target_file, classification.0, classification.1, classification.2, classification.3
                ),
                commit_sha: None,
            });
            st.counters.tests_added += 1;
            st.test_classification.unit += classification.0;
            st.test_classification.integration += classification.1;
            st.test_classification.e2e += classification.2;
            st.test_classification.other += classification.3;
            let last_action = st.actions.last().unwrap().clone();
            let counters = st.counters.clone();
            let tc = st.test_classification.clone();
            st.broadcast_sse_json("action", &last_action);
            st.broadcast_sse_json("counter_update", &counters);
            st.broadcast_sse_json("test_classification", &tc);
            drop(st);

            if !dry_run {
                pr::git_add_commit_push(clone_path, "[agent-pr] feat: add missing tests")?;
            } else {
                println!("[dry-run] Skipping commit/push for added tests");
            }
        }
    }

    Ok(())
}

fn extract_file_path(code: &str, source_files: &[String], project_root: &str) -> Option<String> {
    let first_line = code.lines().next().unwrap_or("");
    if first_line.contains("FILE:") {
        let path = first_line.split("FILE:").nth(1)?.trim().to_string();
        if path.starts_with('/') {
            return Some(path);
        }
        return Some(format!("{}/{}", project_root, path));
    }
    source_files.first().map(|s| s.to_string())
}

fn remove_file_path_line(code: &str) -> String {
    let first_line = code.lines().next().unwrap_or("");
    if first_line.contains("FILE:") {
        code.lines().skip(1).collect::<Vec<_>>().join("\n")
    } else {
        code.to_string()
    }
}

fn parse_test_classification(response: &str) -> (u64, u64, u64, u64) {
    for line in response.lines() {
        if line.starts_with("TEST_TYPE:") {
            let rest = line.trim_start_matches("TEST_TYPE:").trim();
            let mut unit = 0u64;
            let mut integration = 0u64;
            let mut e2e = 0u64;
            let mut other = 0u64;
            for part in rest.split(',') {
                let part = part.trim();
                if let Some((key, val)) = part.split_once('=') {
                    let n = val.trim().parse::<u64>().unwrap_or(0);
                    match key.trim() {
                        "unit" => unit = n,
                        "integration" => integration = n,
                        "e2e" => e2e = n,
                        _ => other += n,
                    }
                }
            }
            return (unit, integration, e2e, other);
        }
    }
    (0, 0, 0, 1)
}
