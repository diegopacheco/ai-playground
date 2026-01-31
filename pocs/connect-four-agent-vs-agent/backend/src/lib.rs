pub mod routes;
pub mod game;
pub mod agents;
pub mod persistence;
pub mod sse;

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::persistence::db::Database;
use crate::sse::broadcaster::Broadcaster;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub broadcaster: Arc<RwLock<Broadcaster>>,
}
