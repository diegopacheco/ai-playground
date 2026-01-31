use serde::Serialize;
use std::collections::HashMap;
use tokio::sync::broadcast;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GameEvent {
    BoardUpdate {
        board: [[char; 7]; 6],
        current_player: String,
        last_move: Option<u8>,
    },
    AgentThinking {
        agent: String,
    },
    AgentMoved {
        agent: String,
        column: u8,
    },
    GameOver {
        winner: Option<String>,
        is_draw: bool,
        duration_ms: u64,
    },
    Error {
        message: String,
    },
}

pub struct Broadcaster {
    channels: HashMap<String, broadcast::Sender<GameEvent>>,
}

impl Broadcaster {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    pub fn create_channel(&mut self, game_id: &str) -> broadcast::Receiver<GameEvent> {
        let (tx, rx) = broadcast::channel(100);
        self.channels.insert(game_id.to_string(), tx);
        rx
    }

    pub fn subscribe(&self, game_id: &str) -> Option<broadcast::Receiver<GameEvent>> {
        self.channels.get(game_id).map(|tx| tx.subscribe())
    }

    pub fn send(&self, game_id: &str, event: GameEvent) {
        if let Some(tx) = self.channels.get(game_id) {
            let _ = tx.send(event);
        }
    }

    pub fn remove_channel(&mut self, game_id: &str) {
        self.channels.remove(game_id);
    }
}

impl Default for Broadcaster {
    fn default() -> Self {
        Self::new()
    }
}
