use tokio::sync::broadcast;

#[derive(Clone)]
pub struct Broadcaster {
    sender: broadcast::Sender<String>,
}

impl Broadcaster {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(100);
        Self { sender }
    }

    pub fn send(&self, msg: &str) {
        let _ = self.sender.send(msg.to_string());
    }

    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.sender.subscribe()
    }
}
