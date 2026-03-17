use sqlx::PgPool;
use sqlx::Row;

pub async fn init_db(pool: &PgPool) {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS query_history (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            generated_sql TEXT,
            result TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )"
    )
    .execute(pool)
    .await
    .expect("Failed to create query_history table");
}

pub async fn create_query(pool: &PgPool, id: &str, question: &str) {
    sqlx::query("INSERT INTO query_history (id, question, status) VALUES ($1, $2, 'pending')")
        .bind(id)
        .bind(question)
        .execute(pool)
        .await
        .expect("Failed to insert query");
}

pub async fn update_query_result(pool: &PgPool, id: &str, status: &str, sql: &str, result: &str) {
    sqlx::query("UPDATE query_history SET status = $1, generated_sql = $2, result = $3 WHERE id = $4")
        .bind(status)
        .bind(sql)
        .bind(result)
        .bind(id)
        .execute(pool)
        .await
        .expect("Failed to update query result");
}

pub async fn get_all_queries(pool: &PgPool) -> Vec<super::models::QueryRecord> {
    let rows = sqlx::query("SELECT id, question, status, generated_sql, result, created_at::text as created_at FROM query_history ORDER BY created_at DESC")
        .fetch_all(pool)
        .await
        .unwrap_or_default();

    rows.iter().map(|r| super::models::QueryRecord {
        id: r.get("id"),
        question: r.get("question"),
        status: r.get("status"),
        generated_sql: r.get("generated_sql"),
        result: r.get("result"),
        created_at: r.get("created_at"),
    }).collect()
}

pub async fn get_query(pool: &PgPool, id: &str) -> Option<super::models::QueryRecord> {
    let row = sqlx::query("SELECT id, question, status, generated_sql, result, created_at::text as created_at FROM query_history WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await
        .ok()?;

    row.map(|r| super::models::QueryRecord {
        id: r.get("id"),
        question: r.get("question"),
        status: r.get("status"),
        generated_sql: r.get("generated_sql"),
        result: r.get("result"),
        created_at: r.get("created_at"),
    })
}
