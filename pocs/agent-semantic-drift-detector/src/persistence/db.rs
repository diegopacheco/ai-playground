use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use crate::persistence::models::DriftRecord;

pub async fn init_db() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect("sqlite:drift.db?mode=rwc")
        .await
        .expect("Failed to connect to SQLite");

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS drift_records (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"
    )
    .execute(&pool)
    .await
    .expect("Failed to create table");

    pool
}

pub async fn save_record(pool: &SqlitePool, record: &DriftRecord) {
    sqlx::query(
        "INSERT INTO drift_records (id, prompt, response, embedding_json, created_at) VALUES (?, ?, ?, ?, ?)"
    )
    .bind(&record.id)
    .bind(&record.prompt)
    .bind(&record.response)
    .bind(&record.embedding_json)
    .bind(&record.created_at)
    .execute(pool)
    .await
    .expect("Failed to save record");
}

pub async fn get_records_for_prompt(pool: &SqlitePool, prompt: &str) -> Vec<DriftRecord> {
    sqlx::query_as::<_, DriftRecord>(
        "SELECT id, prompt, response, embedding_json, created_at FROM drift_records WHERE prompt = ? ORDER BY created_at ASC"
    )
    .bind(prompt)
    .fetch_all(pool)
    .await
    .expect("Failed to fetch records")
}
