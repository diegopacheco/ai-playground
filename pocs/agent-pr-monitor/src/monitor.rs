use crate::agents;
use crate::compiler;
use crate::commenter;
use crate::pr;
use crate::state::{ActionType, AgentAction, AgentLog, SharedState, now_timestamp};
use crate::tester;
use std::time::Duration;

pub fn run_single_cycle(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64, dry_run: bool, state: &SharedState,
) {
    {
        let mut st = state.lock().unwrap();
        st.counters.total_cycles += 1;
        st.last_check = Some(now_timestamp());
        let cycle = st.counters.total_cycles;
        let counters = st.counters.clone();
        st.broadcast_sse_json("cycle_start", &serde_json::json!({"cycle": cycle}));
        st.broadcast_sse_json("counter_update", &counters);
    }

    println!("[{}] Checking PR #{}...", now_timestamp(), pr_number);

    if !dry_run {
        match pr::git_pull(clone_path) {
            Ok(_) => println!("git pull: OK"),
            Err(e) => {
                if e.contains("Merge conflict") || e.contains("CONFLICT") {
                    println!("Merge conflict detected on pull, attempting resolution...");
                    if let Err(err) = resolve_merge_conflict(clone_path, agent, model, state, dry_run) {
                        println!("Failed to resolve merge conflict: {}", err);
                    }
                } else {
                    println!("git pull error: {}", e);
                }
            }
        }
        let base_branch = pr::get_pr_base_branch(owner, repo, pr_number).unwrap_or_else(|_| "main".to_string());
        match pr::merge_base_branch(clone_path, &base_branch) {
            Ok(_) => println!("Base branch merge: OK"),
            Err(e) => {
                if e.contains("Merge conflict") || e.contains("CONFLICT") {
                    println!("Merge conflict with base branch, attempting resolution...");
                    if let Err(err) = resolve_merge_conflict(clone_path, agent, model, state, dry_run) {
                        println!("Failed to resolve merge conflict: {}", err);
                    }
                } else {
                    println!("Base branch merge error: {}", e);
                }
            }
        }
    } else {
        let is_first_cycle = {
            let st = state.lock().unwrap();
            st.counters.total_cycles == 1
        };
        if is_first_cycle {
            match pr::git_pull(clone_path) {
                Ok(_) => println!("git pull: OK (first cycle)"),
                Err(e) => println!("git pull error: {}", e),
            }
        } else {
            println!("[dry-run] Skipping git pull to preserve local fixes");
        }
    }

    let changed_files = pr::get_pr_changed_files(owner, repo, pr_number).unwrap_or_default();
    if !changed_files.is_empty() {
        let detected = crate::detect::detect_project_from_changed_files(clone_path, &changed_files);
        println!("Project detected: {} at {}", detected.project_type, detected.project_root);
    }

    match compiler::check_and_fix_compilation(clone_path, agent, model, owner, repo, pr_number, state, dry_run) {
        Ok(_) => println!("Compilation: OK"),
        Err(e) => println!("Compilation check: {}", e),
    }

    match tester::check_and_fix_tests(clone_path, agent, model, owner, repo, pr_number, state, dry_run) {
        Ok(_) => println!("Tests: OK"),
        Err(e) => println!("Test check: {}", e),
    }

    match commenter::check_and_reply_comments(clone_path, agent, model, owner, repo, pr_number, state, dry_run) {
        Ok(_) => println!("Comments: OK"),
        Err(e) => println!("Comment check: {}", e),
    }

    {
        let mut st = state.lock().unwrap();
        st.refresh_file_tree_scoped(clone_path, &changed_files);
        let cycle = st.counters.total_cycles;
        st.broadcast_sse_json("cycle_end", &serde_json::json!({"cycle": cycle, "status": "complete"}));
        println!(
            "Cycle {} complete - compile fixes: {}, test fixes: {}, tests added: {}, comments: {}",
            st.counters.total_cycles,
            st.counters.compilation_fixes,
            st.counters.test_fixes,
            st.counters.tests_added,
            st.counters.comments_answered,
        );
    }
}

pub async fn run_monitor_loop(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64, refresh_secs: u64, dry_run: bool, state: SharedState,
) {
    let clone_path = clone_path.to_string();
    let agent = agent.to_string();
    let model = model.to_string();
    let owner = owner.to_string();
    let repo = repo.to_string();

    loop {
        run_single_cycle(&clone_path, &agent, &model, &owner, &repo, pr_number, dry_run, &state);
        tokio::time::sleep(Duration::from_secs(refresh_secs)).await;
    }
}

fn resolve_merge_conflict(
    clone_path: &str, agent: &str, model: &str, state: &SharedState, dry_run: bool,
) -> Result<(), String> {
    let conflicted = pr::get_conflicted_files(clone_path);
    if conflicted.is_empty() {
        return Ok(());
    }

    for file in &conflicted {
        let full_path = format!("{}/{}", clone_path, file);
        let content = pr::read_file(&full_path)?;

        let prompt = format!(
            "You are a senior developer. The following file has merge conflicts.\n\
            Resolve the conflicts and return ONLY the resolved file content inside a code block.\n\
            Keep the best of both changes where possible.\n\n\
            File: {}\nContent with conflict markers:\n{}",
            file, content
        );

        let response = agents::run_llm(agent, model, &prompt)?;

        if let Some(resolved) = agents::extract_code_block(&response) {
            pr::write_file(&full_path, &resolved)?;

            let mut st = state.lock().unwrap();
            let id = st.next_id();
            st.add_action(AgentAction {
                id,
                timestamp: now_timestamp(),
                action_type: ActionType::MergeConflictFix,
                description: format!("Resolved merge conflict in {}", file),
                files_changed: vec![file.clone()],
                llm_agent: agent.to_string(),
                llm_model: model.to_string(),
                commit_sha: None,
            });
            st.add_log(AgentLog {
                id,
                timestamp: now_timestamp(),
                action_type: ActionType::MergeConflictFix,
                llm_agent: agent.to_string(),
                llm_model: model.to_string(),
                prompt,
                response: response.clone(),
                result: format!("Resolved conflict in {}", file),
                commit_sha: None,
            });
            let last_action = st.actions.last().unwrap().clone();
            st.broadcast_sse_json("action", &last_action);
            drop(st);
        }
    }

    if !dry_run {
        pr::git_add_commit_push(clone_path, "[agent-pr] fix: resolve merge conflicts")?;
    } else {
        println!("[dry-run] Skipping commit/push for merge conflict resolution");
    }
    Ok(())
}
