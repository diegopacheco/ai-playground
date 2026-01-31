use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DebateEvent {
    AgentThinking { agent: String },
    AgentMessage { agent: String, content: String, stance: String },
    DebateOver { winner: String, reason: String, duration_ms: u64 },
    Error { message: String },
}

pub struct Broadcaster {
    channels: RwLock<HashMap<String, broadcast::Sender<DebateEvent>>>,
}

impl Broadcaster {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            channels: RwLock::new(HashMap::new()),
        })
    }

    pub async fn create_channel(&self, debate_id: &str) -> broadcast::Receiver<DebateEvent> {
        let mut channels = self.channels.write().await;
        let (tx, rx) = broadcast::channel(100);
        channels.insert(debate_id.to_string(), tx);
        rx
    }

    pub async fn subscribe(&self, debate_id: &str) -> Option<broadcast::Receiver<DebateEvent>> {
        let channels = self.channels.read().await;
        channels.get(debate_id).map(|tx| tx.subscribe())
    }

    pub async fn broadcast(&self, debate_id: &str, event: DebateEvent) {
        let channels = self.channels.read().await;
        if let Some(tx) = channels.get(debate_id) {
            let _ = tx.send(event);
        }
    }

    pub async fn remove_channel(&self, debate_id: &str) {
        let mut channels = self.channels.write().await;
        channels.remove(debate_id);
    }
}
