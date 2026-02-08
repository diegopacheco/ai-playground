use std::sync::Arc;
use sqlx::{Pool, Sqlite};
use crate::sse::broadcaster::Broadcaster;

pub mod agents;
pub mod build;
pub mod persistence;
pub mod routes;
pub mod sse;

#[derive(Clone)]
pub struct AppState {
    pub pool: Pool<Sqlite>,
    pub broadcaster: Arc<Broadcaster>,
}
