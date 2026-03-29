#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ActionType {
    CompileFix,
    TestFix,
    TestAdd,
    CommentReply,
    MergeConflictFix,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrInfo {
    pub url: String,
    pub owner: String,
    pub repo: String,
    pub pr_number: u64,
    pub title: String,
    pub branch: String,
    pub total_files: usize,
    pub clone_path: String,
    pub agent_name: String,
    pub agent_model: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentAction {
    pub id: u64,
    pub timestamp: String,
    pub action_type: ActionType,
    pub description: String,
    pub files_changed: Vec<String>,
    pub llm_agent: String,
    pub llm_model: String,
    pub commit_sha: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Counters {
    pub compilation_fixes: u64,
    pub test_fixes: u64,
    pub tests_added: u64,
    pub comments_answered: u64,
    pub total_cycles: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestClassification {
    pub unit: u64,
    pub integration: u64,
    pub e2e: u64,
    pub other: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommentThread {
    pub id: u64,
    pub github_comment_id: u64,
    pub author: String,
    pub body: String,
    pub file_path: Option<String>,
    pub line: Option<u64>,
    pub timestamp: String,
    pub replies: Vec<CommentReply>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommentReply {
    pub author: String,
    pub body: String,
    pub timestamp: String,
    pub is_agent: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub children: Vec<FileEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentLog {
    pub id: u64,
    pub timestamp: String,
    pub action_type: ActionType,
    pub llm_agent: String,
    pub llm_model: String,
    pub prompt: String,
    pub response: String,
    pub result: String,
    pub commit_sha: Option<String>,
}

pub struct AppState {
    pub pr_info: PrInfo,
    pub actions: Vec<AgentAction>,
    pub counters: Counters,
    pub comments: Vec<CommentThread>,
    pub file_tree: Vec<FileEntry>,
    pub logs: Vec<AgentLog>,
    pub start_time: Instant,
    pub last_check: Option<String>,
    pub answered_comment_ids: Vec<u64>,
    pub sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
    pub dry_run: bool,
    pub test_classification: TestClassification,
}

pub type SharedState = Arc<Mutex<AppState>>;

pub fn now_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = now.as_secs();
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

impl AppState {
    pub fn new(pr_info: PrInfo) -> Self {
        AppState {
            pr_info,
            actions: Vec::new(),
            counters: Counters {
                compilation_fixes: 0,
                test_fixes: 0,
                tests_added: 0,
                comments_answered: 0,
                total_cycles: 0,
            },
            comments: Vec::new(),
            file_tree: Vec::new(),
            logs: Vec::new(),
            start_time: Instant::now(),
            last_check: None,
            answered_comment_ids: Vec::new(),
            sse_clients: Vec::new(),
            dry_run: false,
            test_classification: TestClassification {
                unit: 0,
                integration: 0,
                e2e: 0,
                other: 0,
            },
        }
    }

    pub fn add_action(&mut self, action: AgentAction) {
        self.actions.push(action);
    }

    pub fn add_log(&mut self, log: AgentLog) {
        let json = serde_json::to_string(&log).unwrap_or_default();
        self.logs.push(log);
        let msg = format!("event: new_log\ndata: {}\n\n", json);
        self.sse_clients.retain(|sender| sender.send(msg.clone()).is_ok());
    }

    pub fn add_comment(&mut self, thread: CommentThread) {
        self.comments.push(thread);
    }

    pub fn add_reply(&mut self, github_comment_id: u64, reply: CommentReply) {
        for thread in &mut self.comments {
            if thread.github_comment_id == github_comment_id {
                thread.replies.push(reply);
                return;
            }
        }
    }

    pub fn refresh_file_tree(&mut self) {
        let path = Path::new(&self.pr_info.clone_path);
        if path.exists() {
            self.file_tree = build_file_tree(path);
        }
    }

    pub fn refresh_file_tree_scoped(&mut self, clone_path: &str, changed_files: &[String]) {
        if changed_files.is_empty() {
            self.refresh_file_tree();
            return;
        }
        let dirs = crate::pr::get_changed_dirs(clone_path, changed_files);
        let mut entries = Vec::new();
        for dir in &dirs {
            let path = Path::new(dir);
            if path.exists() {
                let dir_name = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| dir.clone());
                entries.push(FileEntry {
                    path: dir.clone(),
                    name: dir_name,
                    is_dir: true,
                    children: build_file_tree(path),
                });
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        self.file_tree = entries;
    }

    pub fn broadcast_sse(&mut self, event: &str, data: &str) {
        let msg = format!("event: {}\ndata: {}\n\n", event, data);
        self.sse_clients.retain(|sender| sender.send(msg.clone()).is_ok());
    }

    pub fn broadcast_sse_json<T: serde::Serialize>(&mut self, event: &str, data: &T) {
        let json = serde_json::to_string(data).unwrap_or_default();
        let msg = format!("event: {}\ndata: {}\n\n", event, json);
        self.sse_clients.retain(|sender| sender.send(msg.clone()).is_ok());
    }

    pub fn next_id(&self) -> u64 {
        self.actions.len() as u64 + 1
    }
}

fn build_file_tree(path: &Path) -> Vec<FileEntry> {
    let mut entries = Vec::new();
    let read_dir = match fs::read_dir(path) {
        Ok(rd) => rd,
        Err(_) => return entries,
    };
    for entry in read_dir.flatten() {
        let entry_path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name == ".git" || name == "target" || name == "node_modules" {
            continue;
        }
        let is_dir = entry_path.is_dir();
        let children = if is_dir {
            build_file_tree(&entry_path)
        } else {
            Vec::new()
        };
        entries.push(FileEntry {
            path: entry_path.to_string_lossy().to_string(),
            name,
            is_dir,
            children,
        });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}
