use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub broadcaster: Broadcaster,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum QueryEvent {
    Thinking { message: String },
    SqlGenerated { sql: String, attempt: u32 },
    SqlError { error: String, attempt: u32 },
    SqlFixed { sql: String, attempt: u32 },
    QueryResult { columns: Vec<String>, rows: Vec<Vec<serde_json::Value>>, sql: String },
    Failed { error: String },
}

#[derive(Clone)]
pub struct Broadcaster {
    channels: Arc<RwLock<HashMap<String, broadcast::Sender<QueryEvent>>>>,
}

impl Broadcaster {
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_channel(&self, id: &str) -> broadcast::Receiver<QueryEvent> {
        let mut channels = self.channels.write().await;
        let (tx, rx) = broadcast::channel(100);
        channels.insert(id.to_string(), tx);
        rx
    }

    pub async fn send(&self, id: &str, event: QueryEvent) {
        let channels = self.channels.read().await;
        if let Some(tx) = channels.get(id) {
            let _ = tx.send(event);
        }
    }

    pub async fn subscribe(&self, id: &str) -> Option<broadcast::Receiver<QueryEvent>> {
        let channels = self.channels.read().await;
        channels.get(id).map(|tx| tx.subscribe())
    }

    pub async fn remove(&self, id: &str) {
        let mut channels = self.channels.write().await;
        channels.remove(id);
    }
}
