use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

pub async fn create_pool(database_url: &str) -> PgPool {
    PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
        .expect("Failed to create database pool")
}

pub async fn run_migrations(pool: &PgPool) {
    sqlx::raw_sql(
        "CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            display_name VARCHAR(100) NOT NULL,
            bio TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS tweets (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) NOT NULL,
            content VARCHAR(280) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS likes (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) NOT NULL,
            tweet_id INTEGER REFERENCES tweets(id) ON DELETE CASCADE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id, tweet_id)
        );
        CREATE TABLE IF NOT EXISTS follows (
            id SERIAL PRIMARY KEY,
            follower_id INTEGER REFERENCES users(id) NOT NULL,
            following_id INTEGER REFERENCES users(id) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(follower_id, following_id),
            CHECK(follower_id != following_id)
        );"
    )
    .execute(pool)
    .await
    .expect("Failed to run migrations");
}
