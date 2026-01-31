use sqlx::{sqlite::SqlitePool, Pool, Sqlite};
use crate::persistence::models::{DebateRecord, MessageRecord};

pub async fn init_db() -> Pool<Sqlite> {
    let pool = SqlitePool::connect("sqlite:debates.db?mode=rwc")
        .await
        .expect("Failed to connect to database");

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS debates (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            agent_a TEXT NOT NULL,
            agent_b TEXT NOT NULL,
            agent_judge TEXT NOT NULL,
            winner TEXT,
            judge_reason TEXT,
            duration_seconds INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create debates table");

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            content TEXT NOT NULL,
            stance TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (debate_id) REFERENCES debates(id)
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create messages table");

    pool
}

pub async fn create_debate(pool: &Pool<Sqlite>, debate: &DebateRecord) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO debates (id, topic, agent_a, agent_b, agent_judge, winner, judge_reason, duration_seconds, started_at, ended_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&debate.id)
    .bind(&debate.topic)
    .bind(&debate.agent_a)
    .bind(&debate.agent_b)
    .bind(&debate.agent_judge)
    .bind(&debate.winner)
    .bind(&debate.judge_reason)
    .bind(debate.duration_seconds)
    .bind(&debate.started_at)
    .bind(&debate.ended_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_debate_result(
    pool: &Pool<Sqlite>,
    debate_id: &str,
    winner: &str,
    reason: &str,
    ended_at: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        UPDATE debates SET winner = ?, judge_reason = ?, ended_at = ? WHERE id = ?
        "#,
    )
    .bind(winner)
    .bind(reason)
    .bind(ended_at)
    .bind(debate_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn save_message(pool: &Pool<Sqlite>, msg: &MessageRecord) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO messages (debate_id, agent, content, stance, created_at)
        VALUES (?, ?, ?, ?, ?)
        "#,
    )
    .bind(&msg.debate_id)
    .bind(&msg.agent)
    .bind(&msg.content)
    .bind(&msg.stance)
    .bind(&msg.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn get_debate(pool: &Pool<Sqlite>, id: &str) -> Result<Option<DebateRecord>, sqlx::Error> {
    let debate = sqlx::query_as::<_, DebateRecord>(
        r#"SELECT * FROM debates WHERE id = ?"#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(debate)
}

pub async fn get_messages(pool: &Pool<Sqlite>, debate_id: &str) -> Result<Vec<MessageRecord>, sqlx::Error> {
    let messages = sqlx::query_as::<_, MessageRecord>(
        r#"SELECT * FROM messages WHERE debate_id = ? ORDER BY id ASC"#,
    )
    .bind(debate_id)
    .fetch_all(pool)
    .await?;
    Ok(messages)
}

pub async fn get_all_debates(pool: &Pool<Sqlite>) -> Result<Vec<DebateRecord>, sqlx::Error> {
    let debates = sqlx::query_as::<_, DebateRecord>(
        r#"SELECT * FROM debates ORDER BY started_at DESC"#,
    )
    .fetch_all(pool)
    .await?;
    Ok(debates)
}
