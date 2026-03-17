use std::collections::HashMap;
use std::sync::RwLock;
use tokio::sync::broadcast;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum GameEvent {
    DmNarration { text: String },
    DmThinking,
    GameOver { reason: String },
    Error { message: String },
}

pub struct Broadcaster {
    channels: RwLock<HashMap<String, broadcast::Sender<GameEvent>>>,
}

impl Broadcaster {
    pub fn new() -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
        }
    }

    pub fn create_channel(&self, game_id: &str) -> broadcast::Receiver<GameEvent> {
        let (tx, rx) = broadcast::channel(100);
        self.channels.write().unwrap().insert(game_id.to_string(), tx);
        rx
    }

    pub fn subscribe(&self, game_id: &str) -> Option<broadcast::Receiver<GameEvent>> {
        self.channels.read().unwrap().get(game_id).map(|tx| tx.subscribe())
    }

    pub fn send(&self, game_id: &str, event: GameEvent) {
        if let Some(tx) = self.channels.read().unwrap().get(game_id) {
            let _ = tx.send(event);
        }
    }
}
