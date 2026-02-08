use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuildEvent {
    StatusUpdate { step: String, progress: u8 },
    BuildComplete { project_id: String },
    Error { message: String },
}

pub struct Broadcaster {
    channels: RwLock<HashMap<String, broadcast::Sender<BuildEvent>>>,
}

impl Broadcaster {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            channels: RwLock::new(HashMap::new()),
        })
    }

    pub async fn create_channel(&self, project_id: &str) -> broadcast::Receiver<BuildEvent> {
        let mut channels = self.channels.write().await;
        let (tx, rx) = broadcast::channel(100);
        channels.insert(project_id.to_string(), tx);
        rx
    }

    pub async fn subscribe(&self, project_id: &str) -> Option<broadcast::Receiver<BuildEvent>> {
        let channels = self.channels.read().await;
        channels.get(project_id).map(|tx| tx.subscribe())
    }

    pub async fn broadcast(&self, project_id: &str, event: BuildEvent) {
        let channels = self.channels.read().await;
        if let Some(tx) = channels.get(project_id) {
            let _ = tx.send(event);
        }
    }

    pub async fn remove_channel(&self, project_id: &str) {
        let mut channels = self.channels.write().await;
        channels.remove(project_id);
    }
}
