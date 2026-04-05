use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAgentRequest {
    pub name: String,
    pub agent_type: String,
    pub task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AgentRecord {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub task: String,
    pub status: String,
    pub desk_index: i64,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MessageRecord {
    pub id: String,
    pub agent_id: String,
    pub content: String,
    pub role: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentWithMessages {
    #[serde(flatten)]
    pub agent: AgentRecord,
    pub messages: Vec<MessageRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentListItem {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub task: String,
    pub status: String,
    pub desk_index: i64,
}
