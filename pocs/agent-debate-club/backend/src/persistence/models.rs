use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DebateRecord {
    pub id: String,
    pub topic: String,
    pub agent_a: String,
    pub agent_b: String,
    pub agent_judge: String,
    pub winner: Option<String>,
    pub judge_reason: Option<String>,
    pub duration_seconds: i64,
    pub started_at: String,
    pub ended_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct MessageRecord {
    pub id: i64,
    pub debate_id: String,
    pub agent: String,
    pub content: String,
    pub stance: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDebateRequest {
    pub topic: String,
    pub agent_a: String,
    pub agent_b: String,
    pub agent_judge: String,
    pub duration_seconds: i64,
    #[serde(default = "default_style")]
    pub style_a: String,
    #[serde(default = "default_style")]
    pub style_b: String,
}

fn default_style() -> String {
    "neutral".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateResponse {
    pub id: String,
    pub topic: String,
    pub agent_a: String,
    pub agent_b: String,
    pub agent_judge: String,
    pub winner: Option<String>,
    pub judge_reason: Option<String>,
    pub duration_seconds: i64,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub messages: Vec<MessageRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
}
