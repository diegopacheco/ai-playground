use sqlx::SqlitePool;

pub async fn create_pool() -> Result<SqlitePool, sqlx::Error> {
    let pool = SqlitePool::connect("sqlite:twitter.db?mode=rwc").await?;
    create_tables(&pool).await?;
    Ok(pool)
}

async fn create_tables(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS likes (
            user_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (user_id, post_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS follows (
            follower_id TEXT NOT NULL,
            following_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (follower_id, following_id),
            FOREIGN KEY (follower_id) REFERENCES users(id),
            FOREIGN KEY (following_id) REFERENCES users(id)
        )",
    )
    .execute(pool)
    .await?;

    Ok(())
}
