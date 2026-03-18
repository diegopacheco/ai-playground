pub mod agents;
pub mod routes;
pub mod sse;
pub mod otel;

use std::sync::{Arc, Mutex};
use sse::Broadcaster;
use otel::TraceRecord;

#[derive(Clone)]
pub struct AppState {
    pub traces: Arc<Mutex<Vec<TraceRecord>>>,
    pub broadcaster: Broadcaster,
}
