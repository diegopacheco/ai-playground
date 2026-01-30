use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct AnalyzeRequest {
    pub prompt: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct DimensionScores {
    pub quality: u8,
    pub stack_definitions: u8,
    pub clear_goals: u8,
    pub non_obvious_decisions: u8,
    pub security_operations: u8,
    pub overall_effectiveness: u8,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelResult {
    pub model: String,
    pub score: Option<u8>,
    pub recommendations: String,
}

#[derive(Debug, Deserialize)]
pub struct AgentResponse {
    pub score: u8,
    pub recommendations: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ProgressEvent {
    #[serde(rename = "start")]
    Start { total_steps: u8, message: String },
    #[serde(rename = "scores")]
    ScoresReady { scores: DimensionScores, step: u8, message: String },
    #[serde(rename = "agent_start")]
    AgentStart { agent: String, step: u8, message: String },
    #[serde(rename = "agent_done")]
    AgentDone { agent: String, result: ModelResult, step: u8, message: String },
    #[serde(rename = "complete")]
    Complete { scores: DimensionScores, model_results: Vec<ModelResult>, message: String },
}
