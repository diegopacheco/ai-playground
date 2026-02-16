use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer};
use chrono::Utc;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Tweet {
    id: i64,
    username: String,
    content: String,
    likes: i64,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct CreateTweet {
    username: String,
    content: String,
}

struct AppState {
    db: Mutex<Connection>,
}

fn init_db(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            content TEXT NOT NULL,
            likes INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )",
    )
    .expect("Failed to create table");
}

async fn get_tweets(data: web::Data<AppState>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let mut stmt = conn
        .prepare("SELECT id, username, content, likes, created_at FROM tweets ORDER BY id DESC")
        .unwrap();
    let tweets: Vec<Tweet> = stmt
        .query_map([], |row| {
            Ok(Tweet {
                id: row.get(0)?,
                username: row.get(1)?,
                content: row.get(2)?,
                likes: row.get(3)?,
                created_at: row.get(4)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    HttpResponse::Ok().json(tweets)
}

async fn create_tweet(data: web::Data<AppState>, body: web::Json<CreateTweet>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO tweets (username, content, likes, created_at) VALUES (?1, ?2, 0, ?3)",
        (&body.username, &body.content, &now),
    )
    .unwrap();
    let id = conn.last_insert_rowid();
    let tweet = Tweet {
        id,
        username: body.username.clone(),
        content: body.content.clone(),
        likes: 0,
        created_at: now,
    };
    HttpResponse::Created().json(tweet)
}

async fn like_tweet(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let rows = conn
        .execute("UPDATE tweets SET likes = likes + 1 WHERE id = ?1", [id])
        .unwrap();
    if rows == 0 {
        return HttpResponse::NotFound().json(serde_json::json!({"error": "Tweet not found"}));
    }
    let mut stmt = conn
        .prepare("SELECT id, username, content, likes, created_at FROM tweets WHERE id = ?1")
        .unwrap();
    let tweet = stmt
        .query_row([id], |row| {
            Ok(Tweet {
                id: row.get(0)?,
                username: row.get(1)?,
                content: row.get(2)?,
                likes: row.get(3)?,
                created_at: row.get(4)?,
            })
        })
        .unwrap();
    HttpResponse::Ok().json(tweet)
}

async fn delete_tweet(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let rows = conn
        .execute("DELETE FROM tweets WHERE id = ?1", [id])
        .unwrap();
    if rows == 0 {
        return HttpResponse::NotFound().json(serde_json::json!({"error": "Tweet not found"}));
    }
    HttpResponse::Ok().json(serde_json::json!({"deleted": id}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let conn = Connection::open("twitter.db").expect("Failed to open database");
    init_db(&conn);

    let data = web::Data::new(AppState {
        db: Mutex::new(conn),
    });

    println!("Server running at http://localhost:8080");

    HttpServer::new(move || {
        let cors = Cors::permissive();
        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .route("/api/tweets", web::get().to(get_tweets))
            .route("/api/tweets", web::post().to(create_tweet))
            .route("/api/tweets/{id}/like", web::post().to(like_tweet))
            .route("/api/tweets/{id}", web::delete().to(delete_tweet))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
