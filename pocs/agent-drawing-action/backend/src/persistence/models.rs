use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct GuessRecord {
    pub id: String,
    pub engine: String,
    pub guess: Option<String>,
    pub status: String,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateGuessRequest {
    pub engine: String,
    pub image: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateGuessResponse {
    pub id: String,
    pub guess: Option<String>,
    pub engine: String,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineInfo {
    pub id: String,
    pub name: String,
}
