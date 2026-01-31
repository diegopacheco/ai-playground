use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use crate::game::state::GameState;
use crate::persistence::models::MatchRecord;

pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn new(path: &str) -> Result<Self, sqlx::Error> {
        let url = format!("sqlite:{}?mode=rwc", path);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await?;
        let db = Self { pool };
        db.init_schema().await?;
        Ok(db)
    }

    async fn init_schema(&self) -> Result<(), sqlx::Error> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS matches (
                id TEXT PRIMARY KEY,
                agent_a TEXT NOT NULL,
                agent_b TEXT NOT NULL,
                winner TEXT,
                is_draw INTEGER NOT NULL DEFAULT 0,
                moves TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_ms INTEGER
            )"
        )
        .execute(&self.pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_matches_started_at ON matches(started_at DESC)")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn save_match(&self, state: &GameState) -> Result<(), sqlx::Error> {
        let moves_json = serde_json::to_string(&state.moves).unwrap_or_default();
        let started_at = state.started_at.to_rfc3339();
        let ended_at = state.ended_at.map(|t| t.to_rfc3339());
        let duration_ms = state.duration_ms();
        sqlx::query(
            "INSERT INTO matches (id, agent_a, agent_b, winner, is_draw, moves, started_at, ended_at, duration_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&state.id)
        .bind(&state.agent_a)
        .bind(&state.agent_b)
        .bind(&state.winner)
        .bind(state.is_draw)
        .bind(&moves_json)
        .bind(&started_at)
        .bind(&ended_at)
        .bind(duration_ms)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_matches(&self, limit: i64) -> Result<Vec<MatchRecord>, sqlx::Error> {
        let records = sqlx::query_as::<_, MatchRecord>(
            "SELECT id, agent_a, agent_b, winner, is_draw, moves, started_at, ended_at, duration_ms
             FROM matches ORDER BY started_at DESC LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(records)
    }

    pub async fn get_match(&self, id: &str) -> Result<Option<MatchRecord>, sqlx::Error> {
        let record = sqlx::query_as::<_, MatchRecord>(
            "SELECT id, agent_a, agent_b, winner, is_draw, moves, started_at, ended_at, duration_ms
             FROM matches WHERE id = ?"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(record)
    }
}
