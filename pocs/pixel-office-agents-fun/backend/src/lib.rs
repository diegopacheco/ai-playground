pub mod routes;
pub mod agents;
pub mod sse;
pub mod persistence;

use persistence::db::Pool;
use sse::broadcaster::Broadcaster;

pub struct AppState {
    pub pool: Pool,
    pub broadcaster: Broadcaster,
}
