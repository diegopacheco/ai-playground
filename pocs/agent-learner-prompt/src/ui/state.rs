use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use super::models::{ConfigResponse, ProgressEvent, TaskStatus};

pub struct AppState {
    pub base_dir: PathBuf,
    pub config: RwLock<ConfigResponse>,
    pub tasks: RwLock<HashMap<String, TaskStatus>>,
    pub event_sender: broadcast::Sender<ProgressEvent>,
}

impl AppState {
    pub fn new(base_dir: PathBuf, agent: String, model: String, cycles: u32) -> Arc<Self> {
        let (tx, _) = broadcast::channel(100);
        Arc::new(Self {
            base_dir,
            config: RwLock::new(ConfigResponse { agent, model, cycles }),
            tasks: RwLock::new(HashMap::new()),
            event_sender: tx,
        })
    }

    pub async fn get_config(&self) -> ConfigResponse {
        self.config.read().await.clone()
    }

    pub async fn update_config(&self, agent: Option<String>, model: Option<String>, cycles: Option<u32>) {
        let mut cfg = self.config.write().await;
        if let Some(a) = agent {
            cfg.agent = a;
        }
        if let Some(m) = model {
            cfg.model = m;
        }
        if let Some(c) = cycles {
            cfg.cycles = c;
        }
    }

    pub async fn add_task(&self, task_id: String, status: TaskStatus) {
        self.tasks.write().await.insert(task_id, status);
    }

    pub async fn get_task(&self, task_id: &str) -> Option<TaskStatus> {
        self.tasks.read().await.get(task_id).cloned()
    }

    pub async fn update_task(&self, task_id: &str, update: impl FnOnce(&mut TaskStatus)) {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(task_id) {
            update(task);
        }
    }

    pub fn send_event(&self, event: ProgressEvent) {
        let _ = self.event_sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ProgressEvent> {
        self.event_sender.subscribe()
    }
}
