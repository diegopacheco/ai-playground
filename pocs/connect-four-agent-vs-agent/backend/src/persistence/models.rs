use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MatchRecord {
    pub id: String,
    pub agent_a: String,
    pub agent_b: String,
    pub winner: Option<String>,
    pub is_draw: bool,
    pub moves: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub duration_ms: Option<i64>,
}
