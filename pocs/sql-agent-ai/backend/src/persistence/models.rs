use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRecord {
    pub id: String,
    pub question: String,
    pub status: String,
    pub generated_sql: Option<String>,
    pub result: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateQueryRequest {
    pub question: String,
}
