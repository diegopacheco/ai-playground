use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task: String,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub cycles: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponse {
    pub task_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub task_id: String,
    pub status: String,
    pub current_cycle: u32,
    pub total_cycles: u32,
    pub phase: String,
    pub completed: bool,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    pub cycles: Vec<String>,
    pub has_memory: bool,
    pub has_mistakes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDetail {
    pub name: String,
    pub memory: String,
    pub mistakes: String,
    pub prompts: String,
    pub cycles: Vec<CycleInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleInfo {
    pub cycle_number: u32,
    pub has_prompt: bool,
    pub has_output: bool,
    pub has_review: bool,
    pub has_learnings: bool,
    pub has_mistakes: bool,
    pub has_improved_prompt: bool,
    pub learnings_content: String,
    pub mistakes_content: String,
    pub improved_prompt_content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigRequest {
    pub agent: Option<String>,
    pub model: Option<String>,
    pub cycles: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigResponse {
    pub agent: String,
    pub model: String,
    pub cycles: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub task_id: String,
    pub event_type: String,
    pub cycle: u32,
    pub phase: String,
    pub message: String,
}
