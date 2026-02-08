use sqlx::{Pool, Sqlite};

pub mod agents;
pub mod routes;
pub mod persistence;

pub struct AppState {
    pub pool: Pool<Sqlite>,
}
