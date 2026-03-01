use memory_game_backend::{create_router, db};
use sqlx::sqlite::SqlitePoolOptions;

#[tokio::main]
async fn main() {
    let db_path = std::path::Path::new("../db");
    std::fs::create_dir_all(db_path).unwrap();

    let database_url = "sqlite:../db/memory_game.db?mode=rwc";

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await
        .unwrap();

    db::init_db(&pool).await;

    let app = create_router(pool);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
