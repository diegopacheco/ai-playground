use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::broadcast;

pub struct Broadcaster {
    channels: Mutex<HashMap<String, broadcast::Sender<String>>>,
}

impl Broadcaster {
    pub fn new() -> Self {
        Self {
            channels: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_channel(&self, id: &str) -> broadcast::Receiver<String> {
        let mut channels = self.channels.lock().unwrap();
        let (tx, rx) = broadcast::channel(100);
        channels.insert(id.to_string(), tx);
        rx
    }

    pub fn subscribe(&self, id: &str) -> Option<broadcast::Receiver<String>> {
        let channels = self.channels.lock().unwrap();
        channels.get(id).map(|tx| tx.subscribe())
    }

    pub fn broadcast(&self, id: &str, message: String) {
        let channels = self.channels.lock().unwrap();
        if let Some(tx) = channels.get(id) {
            let _ = tx.send(message);
        }
    }

    pub fn remove_channel(&self, id: &str) {
        let mut channels = self.channels.lock().unwrap();
        channels.remove(id);
    }
}
