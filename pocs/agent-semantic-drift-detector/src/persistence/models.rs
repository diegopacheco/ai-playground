use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DriftRecord {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub embedding_json: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub date: String,
    pub cosine_similarity: f64,
    pub drift_detected: bool,
}
