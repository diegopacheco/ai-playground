use crate::config::Config;
use sqlx::PgPool;

pub struct AppState {
    pub db: PgPool,
    pub config: Config,
}

impl AppState {
    pub fn new(db: PgPool, config: Config) -> Self {
        Self { db, config }
    }
}
