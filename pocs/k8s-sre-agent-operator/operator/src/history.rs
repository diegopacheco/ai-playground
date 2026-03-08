use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone, Serialize, Deserialize)]
pub struct HistoryEvent {
    pub timestamp: String,
    pub event_type: String,
    pub summary: String,
    pub details: String,
    pub success: bool,
}

pub type HistoryLog = Arc<Mutex<Vec<HistoryEvent>>>;

pub fn new_history() -> HistoryLog {
    Arc::new(Mutex::new(Vec::new()))
}

pub async fn add_event(history: &HistoryLog, event_type: &str, summary: &str, details: &str, success: bool) {
    let event = HistoryEvent {
        timestamp: chrono::Utc::now().to_rfc3339(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        details: details.to_string(),
        success,
    };
    history.lock().await.push(event);
}
