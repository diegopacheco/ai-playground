pub mod claude;
pub mod codex;
pub mod copilot;
pub mod gemini;

use crate::models::AgentStatus;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AgentState {
    pub name: String,
    pub model: String,
    pub status: AgentStatus,
    pub worktree: PathBuf,
}

pub type SharedAgentState = Arc<Mutex<AgentState>>;

pub fn create_agent_state(name: &str, model: &str, worktree: PathBuf) -> SharedAgentState {
    Arc::new(Mutex::new(AgentState {
        name: name.to_string(),
        model: model.to_string(),
        status: AgentStatus::Pending,
        worktree,
    }))
}
