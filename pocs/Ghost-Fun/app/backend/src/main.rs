use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Result};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

struct AppState {
    db: Mutex<Connection>,
}

#[derive(Serialize, Deserialize)]
struct Score {
    id: Option<i64>,
    player_name: String,
    moves: i32,
    time_taken: i32,
    created_at: Option<String>,
}

#[derive(Deserialize)]
struct SubmitScore {
    player_name: String,
    moves: i32,
    time_taken: i32,
}

fn init_db(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            moves INTEGER NOT NULL,
            time_taken INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );",
    ).unwrap();
}

async fn get_leaderboard(data: web::Data<AppState>) -> Result<HttpResponse> {
    let db = data.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT id, player_name, moves, time_taken, created_at FROM scores ORDER BY moves ASC, time_taken ASC LIMIT 10"
    ).unwrap();
    let scores: Vec<Score> = stmt.query_map([], |row| {
        Ok(Score {
            id: Some(row.get(0)?),
            player_name: row.get(1)?,
            moves: row.get(2)?,
            time_taken: row.get(3)?,
            created_at: Some(row.get(4)?),
        })
    }).unwrap().filter_map(|s| s.ok()).collect();
    Ok(HttpResponse::Ok().json(scores))
}

async fn submit_score(data: web::Data<AppState>, body: web::Json<SubmitScore>) -> Result<HttpResponse> {
    let db = data.db.lock().unwrap();
    db.execute(
        "INSERT INTO scores (player_name, moves, time_taken) VALUES (?1, ?2, ?3)",
        params![body.player_name, body.moves, body.time_taken],
    ).unwrap();
    Ok(HttpResponse::Ok().json(serde_json::json!({"status": "ok"})))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let conn = Connection::open("memory.db").unwrap();
    init_db(&conn);
    let data = web::Data::new(AppState {
        db: Mutex::new(conn),
    });
    println!("Backend running on http://localhost:8080");
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();
        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .route("/api/leaderboard", web::get().to(get_leaderboard))
            .route("/api/scores", web::post().to(submit_score))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
