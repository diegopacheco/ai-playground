use crate::models::types::SseEvent;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};

pub async fn create_channel(
    channels: &Arc<Mutex<HashMap<String, broadcast::Sender<String>>>>,
    analysis_id: &str,
) -> broadcast::Sender<String> {
    let (tx, _) = broadcast::channel(100);
    let mut map = channels.lock().await;
    map.insert(analysis_id.to_string(), tx.clone());
    tx
}

pub async fn subscribe(
    channels: &Arc<Mutex<HashMap<String, broadcast::Sender<String>>>>,
    analysis_id: &str,
) -> Option<broadcast::Receiver<String>> {
    let map = channels.lock().await;
    map.get(analysis_id).map(|tx| tx.subscribe())
}

pub fn send_event(sender: &broadcast::Sender<String>, event: SseEvent) {
    let _ = sender.send(event.to_sse_string());
}

pub async fn remove_channel(
    channels: &Arc<Mutex<HashMap<String, broadcast::Sender<String>>>>,
    analysis_id: &str,
) {
    let mut map = channels.lock().await;
    map.remove(analysis_id);
}
