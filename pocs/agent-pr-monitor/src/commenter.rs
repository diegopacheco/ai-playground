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

        {
            let st = state.lock().unwrap();
            if st.answered_comment_ids.contains(&comment_id) {
                continue;
            }
        }

        let code_context = if let Some(ref fp) = file_path {
            let full_path = format!("{}/{}", clone_path, fp);
            pr::read_file(&full_path).unwrap_or_default()
        } else {
            String::new()
        };

        let prompt = format!(
            "You are a developer working on this PR. Reply to the following review comment.\n\
            If the comment requests a code change, explain what you would change.\n\
            If it is a question, provide a clear answer.\n\
            Keep your response concise and helpful.\n\n\
            Comment by @{}: {}\nFile: {}\nCode context:\n{}",
            author, body,
            file_path.as_deref().unwrap_or("N/A"),
            code_context
        );

        let response = agents::run_llm(agent, model, &prompt)?;
        let prefixed = format!("[{}-{}] {}", agent, model, response);

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
            description: format!("Replied to comment by @{}", author),
            files_changed: file_path.into_iter().collect(),
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
            prompt,
            response: response.clone(),
            result: format!("Replied to review comment {}", comment_id),
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

        {
            let st = state.lock().unwrap();
            if st.answered_comment_ids.contains(&comment_id) {
                continue;
            }
        }

        let source_files = pr::list_source_files(clone_path);
        let mut file_context = String::new();
        for f in source_files.iter().take(5) {
            if let Ok(content) = pr::read_file(f) {
                file_context.push_str(&format!("File: {}\n{}\n\n", f, content));
            }
        }

        let prompt = format!(
            "You are a developer working on this PR. Reply to the following review comment.\n\
            If the comment requests a code change, explain what you would change.\n\
            If it is a question, provide a clear answer.\n\
            Keep your response concise and helpful.\n\n\
            Comment by @{}: {}\nFile: N/A\nCode context:\n{}",
            author, body, file_context
        );

        let response = agents::run_llm(agent, model, &prompt)?;
        let prefixed = format!("[{}-{}] {}", agent, model, response);

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
