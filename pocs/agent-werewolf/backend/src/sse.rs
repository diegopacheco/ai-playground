use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::broadcast;

pub struct Broadcaster {
    channels: Mutex<HashMap<String, broadcast::Sender<String>>>,
}

impl Broadcaster {
    pub fn new() -> Self {
        Broadcaster {
            channels: Mutex::new(HashMap::new()),
        }
    }

    pub fn subscribe(&self, game_id: &str) -> broadcast::Receiver<String> {
        let mut channels = self.channels.lock().unwrap();
        let sender = channels.entry(game_id.to_string()).or_insert_with(|| {
            let (tx, _) = broadcast::channel(100);
            tx
        });
        sender.subscribe()
    }

    pub fn send(&self, game_id: &str, event: &str, data: &serde_json::Value) {
        let channels = self.channels.lock().unwrap();
        if let Some(sender) = channels.get(game_id) {
            let msg = format!("event: {}\ndata: {}\n\n", event, serde_json::to_string(data).unwrap_or_default());
            let _ = sender.send(msg);
        }
    }
}
