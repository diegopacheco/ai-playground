use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Game {
    pub id: String,
    pub status: String,
    pub winner: Option<String>,
    pub werewolf_agent: Option<String>,
    pub deception_score: i32,
    pub created_at: String,
    pub ended_at: Option<String>,
    pub agents: Vec<GameAgent>,
    pub rounds: Vec<Round>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAgent {
    pub id: String,
    pub game_id: String,
    pub agent_name: String,
    pub model: String,
    pub role: String,
    pub alive: bool,
    pub votes_correct: i32,
    pub votes_total: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Round {
    pub id: String,
    pub game_id: String,
    pub round_number: i32,
    pub phase: String,
    pub eliminated_agent: Option<String>,
    pub eliminated_by: Option<String>,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub round_id: String,
    pub agent_name: String,
    pub message_type: String,
    pub content: String,
    pub target: Option<String>,
    pub raw_output: Option<String>,
    pub response_time_ms: Option<i64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateGameRequest {
    pub agents: Vec<AgentSelection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSelection {
    pub name: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub name: String,
    pub models: Vec<String>,
    pub default_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub output: String,
    pub elapsed_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NightAction {
    pub target: String,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscussionAction {
    pub statement: String,
    pub suspect: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteAction {
    pub vote: String,
    pub reasoning: String,
}
