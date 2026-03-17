use sqlx::SqlitePool;

pub async fn init_db(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            player_name TEXT NOT NULL,
            setting TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL
        )"
    ).execute(pool).await.unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games(id)
        )"
    ).execute(pool).await.unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS characters (
            game_id TEXT PRIMARY KEY,
            hp INTEGER NOT NULL DEFAULT 100,
            max_hp INTEGER NOT NULL DEFAULT 100,
            level INTEGER NOT NULL DEFAULT 1,
            xp INTEGER NOT NULL DEFAULT 0,
            gold INTEGER NOT NULL DEFAULT 10,
            inventory TEXT NOT NULL DEFAULT '[]',
            location TEXT NOT NULL DEFAULT 'Unknown',
            FOREIGN KEY (game_id) REFERENCES games(id)
        )"
    ).execute(pool).await.unwrap();
}

pub async fn create_game(pool: &SqlitePool, id: &str, player_name: &str, setting: &str) {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("INSERT INTO games (id, player_name, setting, created_at) VALUES (?, ?, ?, ?)")
        .bind(id).bind(player_name).bind(setting).bind(&now)
        .execute(pool).await.unwrap();

    sqlx::query("INSERT INTO characters (game_id) VALUES (?)")
        .bind(id)
        .execute(pool).await.unwrap();
}

pub async fn save_message(pool: &SqlitePool, game_id: &str, role: &str, content: &str) {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("INSERT INTO messages (game_id, role, content, created_at) VALUES (?, ?, ?, ?)")
        .bind(game_id).bind(role).bind(content).bind(&now)
        .execute(pool).await.unwrap();
}

pub async fn get_messages(pool: &SqlitePool, game_id: &str) -> Vec<(String, String)> {
    sqlx::query_as::<_, (String, String)>(
        "SELECT role, content FROM messages WHERE game_id = ? ORDER BY id ASC"
    )
    .bind(game_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default()
}

pub async fn get_character(pool: &SqlitePool, game_id: &str) -> Option<CharacterRow> {
    sqlx::query_as::<_, CharacterRow>(
        "SELECT game_id, hp, max_hp, level, xp, gold, inventory, location FROM characters WHERE game_id = ?"
    )
    .bind(game_id)
    .fetch_optional(pool)
    .await
    .unwrap_or(None)
}

pub async fn update_character(pool: &SqlitePool, game_id: &str, hp: i32, xp: i32, gold: i32, level: i32, inventory: &str, location: &str) {
    sqlx::query(
        "UPDATE characters SET hp = ?, xp = ?, gold = ?, level = ?, inventory = ?, location = ? WHERE game_id = ?"
    )
    .bind(hp).bind(xp).bind(gold).bind(level).bind(inventory).bind(location).bind(game_id)
    .execute(pool).await.unwrap();
}

#[derive(sqlx::FromRow, Clone)]
pub struct CharacterRow {
    #[allow(dead_code)]
    pub game_id: String,
    pub hp: i32,
    pub max_hp: i32,
    pub level: i32,
    pub xp: i32,
    pub gold: i32,
    pub inventory: String,
    pub location: String,
}

pub async fn get_games(pool: &SqlitePool) -> Vec<GameRow> {
    sqlx::query_as::<_, GameRow>(
        "SELECT id, player_name, setting, status, created_at FROM games ORDER BY created_at DESC"
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default()
}

pub async fn get_game(pool: &SqlitePool, id: &str) -> Option<GameRow> {
    sqlx::query_as::<_, GameRow>(
        "SELECT id, player_name, setting, status, created_at FROM games WHERE id = ?"
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .unwrap_or(None)
}

#[derive(sqlx::FromRow, Clone, serde::Serialize)]
pub struct GameRow {
    pub id: String,
    pub player_name: String,
    pub setting: String,
    pub status: String,
    pub created_at: String,
}
