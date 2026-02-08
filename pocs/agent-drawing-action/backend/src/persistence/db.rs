use sqlx::{sqlite::SqlitePool, Pool, Sqlite};
use crate::persistence::models::GuessRecord;

pub async fn init_db() -> Pool<Sqlite> {
    let pool = SqlitePool::connect("sqlite:guesses.db?mode=rwc")
        .await
        .expect("Failed to connect to database");

    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS guesses (
            id TEXT PRIMARY KEY,
            engine TEXT NOT NULL,
            guess TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
        )"#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create guesses table");

    pool
}

pub async fn create_guess(pool: &Pool<Sqlite>, record: &GuessRecord) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"INSERT INTO guesses (id, engine, guess, status, created_at, completed_at, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)"#,
    )
    .bind(&record.id)
    .bind(&record.engine)
    .bind(&record.guess)
    .bind(&record.status)
    .bind(&record.created_at)
    .bind(&record.completed_at)
    .bind(&record.error)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_guess(
    pool: &Pool<Sqlite>,
    id: &str,
    guess: Option<&str>,
    status: &str,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    let completed_at = if status == "done" || status == "error" {
        Some(chrono::Utc::now().to_rfc3339())
    } else {
        None
    };

    sqlx::query(
        r#"UPDATE guesses SET guess = ?, status = ?, error = ?, completed_at = ? WHERE id = ?"#,
    )
    .bind(guess)
    .bind(status)
    .bind(error)
    .bind(completed_at)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn get_guess(pool: &Pool<Sqlite>, id: &str) -> Result<Option<GuessRecord>, sqlx::Error> {
    let record = sqlx::query_as::<_, GuessRecord>(
        r#"SELECT * FROM guesses WHERE id = ?"#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(record)
}

pub async fn get_all_guesses(pool: &Pool<Sqlite>) -> Result<Vec<GuessRecord>, sqlx::Error> {
    let records = sqlx::query_as::<_, GuessRecord>(
        r#"SELECT * FROM guesses ORDER BY created_at DESC"#,
    )
    .fetch_all(pool)
    .await?;
    Ok(records)
}

pub async fn delete_guess(pool: &Pool<Sqlite>, id: &str) -> Result<(), sqlx::Error> {
    sqlx::query(r#"DELETE FROM guesses WHERE id = ?"#)
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}
