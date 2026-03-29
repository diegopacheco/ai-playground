use crate::agents;
use crate::pr;
use crate::state::{
    ActionType, AgentAction, AgentLog, CommentReply, CommentThread, SharedState, now_timestamp,
};

pub fn check_and_reply_comments(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64, state: &SharedState, dry_run: bool,
) -> Result<(), String> {
    handle_review_comments(clone_path, agent, model, owner, repo, pr_number, state, dry_run)?;
    handle_issue_comments(clone_path, agent, model, owner, repo, pr_number, state, dry_run)?;
    Ok(())
}

fn extract_file_path_from_block(code: &str, project_root: &str) -> Option<String> {
    let first_line = code.lines().next().unwrap_or("");
    if first_line.contains("FILE:") {
        let path = first_line.split("FILE:").nth(1)?.trim().to_string();
        if path.starts_with('/') {
            return Some(path);
        }
        return Some(format!("{}/{}", project_root, path));
    }
    None
}

fn remove_file_path_line(code: &str) -> String {
    let first_line = code.lines().next().unwrap_or("");
    if first_line.contains("FILE:") {
        code.lines().skip(1).collect::<Vec<_>>().join("\n")
    } else {
        code.to_string()
    }
}

fn implement_comment_changes(
    clone_path: &str, agent: &str, model: &str,
    _owner: &str, _repo: &str, _pr_number: u64,
    comment_body: &str, comment_author: &str,
    file_context: &str, _state: &SharedState, dry_run: bool,
) -> Result<(String, Vec<String>), String> {
    let prompt = format!(
        "You are a senior developer working on this PR.\n\
        A reviewer has requested changes. You MUST implement them.\n\
        Return the COMPLETE file content for each file you create or modify inside code blocks.\n\
        Each code block MUST have the file path as the first line like: // FILE: path/to/file\n\
        After all code blocks, write a brief SUMMARY line starting with SUMMARY: explaining what you did.\n\n\
        Comment by @{}: {}\n\nProject root: {}\nExisting files:\n{}",
        comment_author, comment_body, clone_path, file_context
    );

    let response = agents::run_llm(agent, model, &prompt)?;
    let blocks = agents::extract_all_code_blocks(&response);
    let mut files_changed = Vec::new();

    for block in &blocks {
        if let Some(target_file) = extract_file_path_from_block(block, clone_path) {
            let clean_code = remove_file_path_line(block);
            if let Some(parent) = std::path::Path::new(&target_file).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            pr::write_file(&target_file, &clean_code)?;
            files_changed.push(target_file);
        }
    }

    if !files_changed.is_empty() && !dry_run {
        let commit_msg = format!("[agent-pr] feat: implement changes requested by @{}", comment_author);
        pr::git_add_commit_push(clone_path, &commit_msg)?;
    } else if !files_changed.is_empty() {
        println!("[dry-run] Skipping commit/push for {} changed files", files_changed.len());
    }

    let summary = response.lines()
        .find(|l| l.starts_with("SUMMARY:"))
        .map(|l| l.trim_start_matches("SUMMARY:").trim().to_string())
        .unwrap_or_else(|| format!("Implemented changes: {} files modified/created", files_changed.len()));

    let reply = if files_changed.is_empty() {
        response.lines()
            .filter(|l| !l.starts_with("```") && !l.contains("FILE:"))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string()
    } else {
        let file_list: Vec<String> = files_changed.iter()
            .map(|f| format!("- `{}`", f.replace(clone_path, "").trim_start_matches('/')))
            .collect();
        format!("{}\n\nFiles changed:\n{}", summary, file_list.join("\n"))
    };

    Ok((reply, files_changed))
}

fn is_agent_comment(body: &str) -> bool {
    body.starts_with("[claude-") || body.starts_with("[gemini-")
        || body.starts_with("[copilot-") || body.starts_with("[codex-")
}

fn handle_review_comments(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64, state: &SharedState, dry_run: bool,
) -> Result<(), String> {
    let json_str = pr::get_pr_comments(owner, repo, pr_number)?;
    let comments: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap_or_default();

    for comment in comments {
        let comment_id = comment["id"].as_u64().unwrap_or(0);
        let author = comment["user"]["login"].as_str().unwrap_or("unknown").to_string();
        let body = comment["body"].as_str().unwrap_or("").to_string();
        let file_path = comment["path"].as_str().map(|s| s.to_string());
        let line = comment["line"].as_u64();

        if is_agent_comment(&body) {
            continue;
        }

        {
            let st = state.lock().unwrap();
            if st.answered_comment_ids.contains(&comment_id) {
                continue;
            }
        }

        let code_context = if let Some(ref fp) = file_path {
            let full_path = format!("{}/{}", clone_path, fp);
            match pr::read_file(&full_path) {
                Ok(content) => format!("File: {}\n{}\n\n", fp, content),
                Err(_) => String::new(),
            }
        } else {
            String::new()
        };

        let (reply, files_changed) = implement_comment_changes(
            clone_path, agent, model, owner, repo, pr_number,
            &body, &author, &code_context, state, dry_run,
        )?;

        let prefixed = format!("[{}-{}] {}", agent, model, reply);

        if !dry_run {
            pr::reply_to_review_comment(owner, repo, pr_number, comment_id, &prefixed)?;
        } else {
            println!("[dry-run] Skipping reply to review comment {}", comment_id);
        }

        let mut st = state.lock().unwrap();
        let id = st.next_id();
        st.add_comment(CommentThread {
            id,
            github_comment_id: comment_id,
            author: author.clone(),
            body: body.clone(),
            file_path: file_path.clone(),
            line,
            timestamp: now_timestamp(),
            replies: vec![CommentReply {
                author: agent.to_string(),
                body: prefixed.clone(),
                timestamp: now_timestamp(),
                is_agent: true,
            }],
        });
        st.add_action(AgentAction {
            id,
            timestamp: now_timestamp(),
            action_type: ActionType::CommentReply,
            description: format!("Implemented changes requested by @{}", author),
            files_changed: files_changed.clone(),
            llm_agent: agent.to_string(),
            llm_model: model.to_string(),
            commit_sha: None,
        });
        st.add_log(AgentLog {
            id,
            timestamp: now_timestamp(),
            action_type: ActionType::CommentReply,
            llm_agent: agent.to_string(),
            llm_model: model.to_string(),
            prompt: format!("Comment by @{}: {}", author, body),
            response: reply.clone(),
            result: format!("Changed {} files for review comment {}", files_changed.len(), comment_id),
            commit_sha: None,
        });
        st.counters.comments_answered += 1;
        st.answered_comment_ids.push(comment_id);
        let last_action = st.actions.last().unwrap().clone();
        let last_comment = st.comments.last().unwrap().clone();
        let counters = st.counters.clone();
        st.broadcast_sse_json("action", &last_action);
        st.broadcast_sse_json("new_comment", &last_comment);
        st.broadcast_sse_json("counter_update", &counters);
        drop(st);
    }

    Ok(())
}

fn handle_issue_comments(
    clone_path: &str, agent: &str, model: &str,
    owner: &str, repo: &str, pr_number: u64, state: &SharedState, dry_run: bool,
) -> Result<(), String> {
    let json_str = pr::get_pr_review_comments(owner, repo, pr_number)?;
    let comments: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap_or_default();

    for comment in comments {
        let comment_id = comment["id"].as_u64().unwrap_or(0);
        let author = comment["user"]["login"].as_str().unwrap_or("unknown").to_string();
        let body = comment["body"].as_str().unwrap_or("").to_string();

        if is_agent_comment(&body) {
            continue;
        }

        {
            let st = state.lock().unwrap();
            if st.answered_comment_ids.contains(&comment_id) {
                continue;
            }
        }

        let source_files = pr::list_source_files(clone_path);
        let mut file_context = String::new();
        for f in source_files.iter().take(10) {
            if let Ok(content) = pr::read_file(f) {
                file_context.push_str(&format!("File: {}\n{}\n\n", f, content));
            }
        }

        let (reply, files_changed) = implement_comment_changes(
            clone_path, agent, model, owner, repo, pr_number,
            &body, &author, &file_context, state, dry_run,
        )?;

        let prefixed = format!("[{}-{}] {}", agent, model, reply);

        if !dry_run {
            pr::post_pr_comment(owner, repo, pr_number, &prefixed)?;
        } else {
            println!("[dry-run] Skipping post of issue comment reply");
        }

        let mut st = state.lock().unwrap();
        let id = st.next_id();
        st.add_comment(CommentThread {
            id,
            github_comment_id: comment_id,
            author: author.clone(),
            body: body.clone(),
            file_path: None,
            line: None,
            timestamp: now_timestamp(),
            replies: vec![CommentReply {
                author: agent.to_string(),
                body: prefixed.clone(),
                timestamp: now_timestamp(),
                is_agent: true,
            }],
        });
        st.add_action(AgentAction {
            id,
            timestamp: now_timestamp(),
            action_type: ActionType::CommentReply,
            description: format!("Implemented changes requested by @{}", author),
            files_changed: files_changed.clone(),
            llm_agent: agent.to_string(),
            llm_model: model.to_string(),
            commit_sha: None,
        });
        st.add_log(AgentLog {
            id,
            timestamp: now_timestamp(),
            action_type: ActionType::CommentReply,
            llm_agent: agent.to_string(),
            llm_model: model.to_string(),
            prompt: format!("Comment by @{}: {}", author, body),
            response: reply.clone(),
            result: format!("Changed {} files for issue comment {}", files_changed.len(), comment_id),
            commit_sha: None,
        });
        st.counters.comments_answered += 1;
        st.answered_comment_ids.push(comment_id);
        let last_comment = st.comments.last().unwrap().clone();
        let counters = st.counters.clone();
        st.broadcast_sse_json("new_comment", &last_comment);
        st.broadcast_sse_json("counter_update", &counters);
        drop(st);
    }

    Ok(())
}
