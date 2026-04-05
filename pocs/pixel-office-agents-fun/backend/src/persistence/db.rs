use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use crate::persistence::models::{AgentRecord, MessageRecord};

pub type Pool = SqlitePool;

pub async fn init_db() -> Result<Pool, sqlx::Error> {
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect("sqlite:pixel_office.db?mode=rwc")
        .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            task TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'spawning',
            desk_index INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT
        )"
    ).execute(&pool).await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            content TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        )"
    ).execute(&pool).await?;

    Ok(pool)
}

pub async fn create_agent(pool: &Pool, agent: &AgentRecord) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO agents (id, name, agent_type, task, status, desk_index, created_at, completed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&agent.id)
    .bind(&agent.name)
    .bind(&agent.agent_type)
    .bind(&agent.task)
    .bind(&agent.status)
    .bind(agent.desk_index)
    .bind(&agent.created_at)
    .bind(&agent.completed_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_agent_status(pool: &Pool, id: &str, status: &str, completed_at: Option<&str>) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE agents SET status = ?, completed_at = ? WHERE id = ?")
        .bind(status)
        .bind(completed_at)
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn save_message(pool: &Pool, msg: &MessageRecord) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO messages (id, agent_id, content, role, created_at) VALUES (?, ?, ?, ?, ?)"
    )
    .bind(&msg.id)
    .bind(&msg.agent_id)
    .bind(&msg.content)
    .bind(&msg.role)
    .bind(&msg.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn clear_all_agents(pool: &Pool) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM messages").execute(pool).await?;
    sqlx::query("DELETE FROM agents").execute(pool).await?;
    Ok(())
}

pub async fn get_all_agents(pool: &Pool) -> Result<Vec<AgentRecord>, sqlx::Error> {
    let agents = sqlx::query_as::<_, AgentRecord>("SELECT * FROM agents ORDER BY created_at DESC")
        .fetch_all(pool)
        .await?;
    Ok(agents)
}

pub async fn get_agent(pool: &Pool, id: &str) -> Result<Option<AgentRecord>, sqlx::Error> {
    let agent = sqlx::query_as::<_, AgentRecord>("SELECT * FROM agents WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    Ok(agent)
}

pub async fn get_messages(pool: &Pool, agent_id: &str) -> Result<Vec<MessageRecord>, sqlx::Error> {
    let messages = sqlx::query_as::<_, MessageRecord>(
        "SELECT * FROM messages WHERE agent_id = ? ORDER BY created_at ASC"
    )
    .bind(agent_id)
    .fetch_all(pool)
    .await?;
    Ok(messages)
}

pub async fn get_next_desk_index(pool: &Pool) -> Result<i64, sqlx::Error> {
    let row: (i64,) = sqlx::query_as(
        "SELECT COALESCE(MAX(desk_index), -1) + 1 FROM agents WHERE status IN ('spawning', 'thinking', 'working')"
    )
    .fetch_one(pool)
    .await
    .unwrap_or((0,));
    Ok(row.0 % 6)
}
