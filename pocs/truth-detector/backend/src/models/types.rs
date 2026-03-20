use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Mutex<rusqlite::Connection>>,
    pub channels: Arc<Mutex<HashMap<String, broadcast::Sender<String>>>>,
}

#[derive(Deserialize)]
pub struct AnalyzeRequest {
    pub github_user: String,
    pub cli: String,
    pub model: String,
}

#[derive(Serialize)]
pub struct AnalyzeResponse {
    pub id: String,
}

#[derive(Serialize, Clone)]
pub struct AgentInfo {
    pub name: String,
    pub models: Vec<String>,
}

pub fn available_agents() -> Vec<AgentInfo> {
    vec![
        AgentInfo { name: "claude".into(), models: vec!["opus".into(), "sonnet".into(), "haiku".into()] },
        AgentInfo { name: "gemini".into(), models: vec!["gemini-3.1-pro".into(), "gemini-3-flash".into(), "gemini-2.5-pro".into()] },
        AgentInfo { name: "copilot".into(), models: vec!["claude-sonnet-4.6".into(), "claude-sonnet-4.5".into(), "gemini-3-pro".into()] },
        AgentInfo { name: "codex".into(), models: vec!["gpt-5.4".into(), "gpt-5.4-mini".into(), "gpt-5.3-codex".into()] },
    ]
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Analysis {
    pub id: String,
    pub github_user: String,
    pub created_at: String,
    pub total_score: Option<i32>,
    pub avg_score: Option<f64>,
    pub status: String,
    pub commits: Vec<CommitResult>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CommitResult {
    pub id: String,
    pub analysis_id: String,
    pub repo_name: String,
    pub commit_sha: String,
    pub commit_message: String,
    pub commit_date: String,
    pub classification: String,
    pub summary: String,
    pub score: i32,
    pub fallback: bool,
    pub raw_llm_output: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WeeklyScore {
    pub id: String,
    pub github_user: String,
    pub week_start: String,
    pub week_end: String,
    pub total_score: i32,
    pub avg_score: f64,
    pub num_commits: i32,
    pub deep_count: i32,
    pub decent_count: i32,
    pub shallow_count: i32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrackedUser {
    pub github_user: String,
    pub added_at: String,
}

#[derive(Deserialize)]
pub struct TrackRequest {
    pub github_user: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LeaderboardEntry {
    pub rank: i32,
    pub github_user: String,
    pub avg_score: f64,
    pub total_score: i32,
    pub num_commits: i32,
    pub deep_count: i32,
    pub decent_count: i32,
    pub shallow_count: i32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LlmCommitResult {
    pub index: usize,
    pub classification: String,
    pub summary: String,
    pub score: i32,
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub struct LlmBatchResponse {
    pub results: Vec<LlmCommitResult>,
}

#[derive(Clone)]
pub enum SseEvent {
    AnalysisStart {
        github_user: String,
        commit_count: usize,
        cached: bool,
    },
    CommitsFetched {
        commits: Vec<SseCommitInfo>,
    },
    LlmThinking,
    AllCommitsAnalyzed {
        results: Vec<SseCommitAnalyzed>,
    },
    AnalysisComplete {
        total_score: i32,
        avg_score: f64,
        deep: i32,
        decent: i32,
        shallow: i32,
    },
    Error {
        message: String,
    },
}

#[derive(Serialize, Clone)]
pub struct SseCommitInfo {
    pub sha: String,
    pub message: String,
    pub repo: String,
}

#[derive(Serialize, Clone)]
pub struct SseCommitAnalyzed {
    pub index: usize,
    pub sha: String,
    pub classification: String,
    pub summary: String,
    pub score: i32,
    pub fallback: bool,
}

impl SseEvent {
    pub fn to_sse_string(&self) -> String {
        match self {
            SseEvent::AnalysisStart {
                github_user,
                commit_count,
                cached,
            } => {
                let data = serde_json::json!({
                    "github_user": github_user,
                    "commit_count": commit_count,
                    "cached": cached,
                });
                format!("event: analysis_start\ndata: {}\n\n", data)
            }
            SseEvent::CommitsFetched { commits } => {
                let data = serde_json::to_string(commits).unwrap_or_default();
                format!("event: commits_fetched\ndata: {}\n\n", data)
            }
            SseEvent::LlmThinking => {
                format!("event: llm_thinking\ndata: {{}}\n\n")
            }
            SseEvent::AllCommitsAnalyzed { results } => {
                let data = serde_json::to_string(results).unwrap_or_default();
                format!("event: all_commits_analyzed\ndata: {}\n\n", data)
            }
            SseEvent::AnalysisComplete {
                total_score,
                avg_score,
                deep,
                decent,
                shallow,
            } => {
                let data = serde_json::json!({
                    "total_score": total_score,
                    "avg_score": avg_score,
                    "deep": deep,
                    "decent": decent,
                    "shallow": shallow,
                });
                format!("event: analysis_complete\ndata: {}\n\n", data)
            }
            SseEvent::Error { message } => {
                let data = serde_json::json!({"message": message});
                format!("event: error\ndata: {}\n\n", data)
            }
        }
    }
}
