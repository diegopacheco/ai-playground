use crate::agents;
use crate::detect::{build_command, detect_project_from_changed_files};
use crate::pr;
use crate::state::{ActionType, AgentAction, AgentLog, SharedState, now_timestamp};
use std::process::Command;

pub fn check_and_fix_compilation(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64,
    state: &SharedState, dry_run: bool,
) -> Result<(), String> {
    let changed_files = pr::get_pr_changed_files(owner, repo, pr_number).unwrap_or_default();
    let detected = detect_project_from_changed_files(clone_path, &changed_files);
    let project_root = &detected.project_root;
    let (cmd, args) = build_command(&detected.project_type);

    println!("Project detected: {} at {}", detected.project_type, project_root);

    let output = Command::new(cmd)
        .args(&args)
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("Build command failed: {}", e))?;

    if output.status.success() {
        return Ok(());
    }

    let mut compiler_error = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    for attempt in 0..10 {
        let source_files = pr::list_changed_source_files(clone_path, &changed_files);
        let mut file_context = String::new();
        for f in &source_files {
            if let Ok(content) = pr::read_file(f) {
                file_context.push_str(&format!("File: {}\n{}\n\n", f, content));
            }
        }

        let prompt = format!(
            "You are a senior developer. The following code has a compilation error.\n\
            Fix the error and return ONLY the corrected file content inside a code block.\n\
            Include the file path as the first line of each code block like: // FILE: path/to/file\n\n\
            Project type: {}\nError:\n{}\n\nSource files:\n{}",
            detected.project_type, compiler_error, file_context
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
                    action_type: ActionType::CompileFix,
                    description: format!("Compilation fix attempt {}", attempt + 1),
                    files_changed: vec![target_file.clone()],
                    llm_agent: agent.to_string(),
                    llm_model: model.to_string(),
                    commit_sha: None,
                });
                st.add_log(AgentLog {
                    id,
                    timestamp: now_timestamp(),
                    action_type: ActionType::CompileFix,
                    llm_agent: agent.to_string(),
                    llm_model: model.to_string(),
                    prompt: prompt.clone(),
                    response: response.clone(),
                    result: format!("Applied fix to {}", target_file),
                    commit_sha: None,
                });
                st.counters.compilation_fixes += 1;
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
            .map_err(|e| format!("Build retry failed: {}", e))?;

        if retry.status.success() {
            if !dry_run {
                pr::git_add_commit_push(clone_path, "[agent-pr] fix: compilation error")?;
            } else {
                println!("[dry-run] Skipping commit/push for compilation fix");
            }
            return Ok(());
        }

        compiler_error = format!(
            "{}{}",
            String::from_utf8_lossy(&retry.stdout),
            String::from_utf8_lossy(&retry.stderr)
        );
    }

    Err("Compilation fix failed after 10 attempts".to_string())
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
