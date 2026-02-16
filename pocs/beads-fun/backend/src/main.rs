use actix_cors::Cors;
use actix_multipart::Multipart;
use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer};
use chrono::Utc;
use futures_util::StreamExt;
use hex;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Tweet {
    id: i64,
    username: String,
    content: String,
    likes: i64,
    image_url: Option<String>,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct CreateTweet {
    content: String,
    image_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RegisterRequest {
    username: String,
    password: String,
}

#[derive(Debug, Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct AuthResponse {
    token: String,
    username: String,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: String,
}

struct AppState {
    db: Mutex<Connection>,
}

fn hash_password(password: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(password.as_bytes());
    hex::encode(hasher.finalize())
}

fn init_db(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            token TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            content TEXT NOT NULL,
            likes INTEGER DEFAULT 0,
            image_url TEXT,
            created_at TEXT NOT NULL
        )",
    )
    .expect("Failed to create tables");
}

fn validate_token(conn: &Connection, req: &HttpRequest) -> Option<String> {
    let header = req.headers().get("Authorization")?.to_str().ok()?;
    let token = header.strip_prefix("Bearer ")?;
    let mut stmt = conn
        .prepare("SELECT username FROM users WHERE token = ?1")
        .ok()?;
    stmt.query_row([token], |row| row.get::<_, String>(0)).ok()
}

async fn register(data: web::Data<AppState>, body: web::Json<RegisterRequest>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let exists: bool = conn
        .prepare("SELECT COUNT(*) FROM users WHERE username = ?1")
        .unwrap()
        .query_row([&body.username], |row| row.get::<_, i64>(0))
        .unwrap()
        > 0;
    if exists {
        return HttpResponse::Conflict()
            .json(serde_json::json!({"error": "Username already taken"}));
    }
    let password_hash = hash_password(&body.password);
    let token = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO users (username, password_hash, token, created_at) VALUES (?1, ?2, ?3, ?4)",
        (&body.username, &password_hash, &token, &now),
    )
    .unwrap();
    HttpResponse::Created().json(AuthResponse {
        token,
        username: body.username.clone(),
    })
}

async fn login(data: web::Data<AppState>, body: web::Json<LoginRequest>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let password_hash = hash_password(&body.password);
    let result = conn
        .prepare("SELECT username FROM users WHERE username = ?1 AND password_hash = ?2")
        .unwrap()
        .query_row([&body.username, &password_hash], |row| {
            row.get::<_, String>(0)
        });
    match result {
        Ok(username) => {
            let token = Uuid::new_v4().to_string();
            conn.execute(
                "UPDATE users SET token = ?1 WHERE username = ?2",
                (&token, &username),
            )
            .unwrap();
            HttpResponse::Ok().json(AuthResponse { token, username })
        }
        Err(_) => {
            HttpResponse::Unauthorized().json(serde_json::json!({"error": "Invalid credentials"}))
        }
    }
}

async fn get_tweets(data: web::Data<AppState>) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let mut stmt = conn
        .prepare("SELECT id, username, content, likes, image_url, created_at FROM tweets ORDER BY id DESC")
        .unwrap();
    let tweets: Vec<Tweet> = stmt
        .query_map([], |row| {
            Ok(Tweet {
                id: row.get(0)?,
                username: row.get(1)?,
                content: row.get(2)?,
                likes: row.get(3)?,
                image_url: row.get(4)?,
                created_at: row.get(5)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    HttpResponse::Ok().json(tweets)
}

async fn create_tweet(
    data: web::Data<AppState>,
    req: HttpRequest,
    body: web::Json<CreateTweet>,
) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let token_user = validate_token(&conn, &req);
    if token_user.is_none() {
        return HttpResponse::Unauthorized()
            .json(serde_json::json!({"error": "Not authenticated"}));
    }
    let username = token_user.unwrap();
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO tweets (username, content, likes, image_url, created_at) VALUES (?1, ?2, 0, ?3, ?4)",
        (&username, &body.content, &body.image_url, &now),
    )
    .unwrap();
    let id = conn.last_insert_rowid();
    let tweet = Tweet {
        id,
        username,
        content: body.content.clone(),
        likes: 0,
        image_url: body.image_url.clone(),
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
        .prepare("SELECT id, username, content, likes, image_url, created_at FROM tweets WHERE id = ?1")
        .unwrap();
    let tweet = stmt
        .query_row([id], |row| {
            Ok(Tweet {
                id: row.get(0)?,
                username: row.get(1)?,
                content: row.get(2)?,
                likes: row.get(3)?,
                image_url: row.get(4)?,
                created_at: row.get(5)?,
            })
        })
        .unwrap();
    HttpResponse::Ok().json(tweet)
}

async fn delete_tweet(
    data: web::Data<AppState>,
    req: HttpRequest,
    path: web::Path<i64>,
) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let token_user = validate_token(&conn, &req);
    if token_user.is_none() {
        return HttpResponse::Unauthorized()
            .json(serde_json::json!({"error": "Not authenticated"}));
    }
    let username = token_user.unwrap();
    let rows = conn
        .execute(
            "DELETE FROM tweets WHERE id = ?1 AND username = ?2",
            rusqlite::params![id, username],
        )
        .unwrap();
    if rows == 0 {
        return HttpResponse::NotFound()
            .json(serde_json::json!({"error": "Tweet not found or not yours"}));
    }
    HttpResponse::Ok().json(serde_json::json!({"deleted": id}))
}

async fn search_tweets(
    data: web::Data<AppState>,
    query: web::Query<SearchQuery>,
) -> HttpResponse {
    let conn = data.db.lock().unwrap();
    let search = format!("%{}%", query.q);
    let mut stmt = conn
        .prepare(
            "SELECT id, username, content, likes, image_url, created_at FROM tweets
             WHERE content LIKE ?1 OR username LIKE ?1
             ORDER BY id DESC",
        )
        .unwrap();
    let tweets: Vec<Tweet> = stmt
        .query_map([&search], |row| {
            Ok(Tweet {
                id: row.get(0)?,
                username: row.get(1)?,
                content: row.get(2)?,
                likes: row.get(3)?,
                image_url: row.get(4)?,
                created_at: row.get(5)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    HttpResponse::Ok().json(tweets)
}

async fn upload_image(mut payload: Multipart) -> HttpResponse {
    while let Some(Ok(mut field)) = payload.next().await {
        let filename = format!("{}.png", Uuid::new_v4());
        let upload_dir = "uploads";
        std::fs::create_dir_all(upload_dir).unwrap();
        let filepath = format!("{}/{}", upload_dir, filename);
        let mut bytes = Vec::new();
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            bytes.extend_from_slice(&data);
        }
        std::fs::write(&filepath, &bytes).unwrap();
        let url = format!("/uploads/{}", filename);
        return HttpResponse::Ok().json(serde_json::json!({"url": url}));
    }
    HttpResponse::BadRequest().json(serde_json::json!({"error": "No file provided"}))
}

async fn serve_upload(path: web::Path<String>) -> HttpResponse {
    let filename = path.into_inner();
    let filepath = format!("uploads/{}", filename);
    match std::fs::read(&filepath) {
        Ok(data) => HttpResponse::Ok().content_type("image/png").body(data),
        Err(_) => HttpResponse::NotFound().json(serde_json::json!({"error": "File not found"})),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let conn = Connection::open("twitter.db").expect("Failed to open database");
    init_db(&conn);
    std::fs::create_dir_all("uploads").unwrap();

    let data = web::Data::new(AppState {
        db: Mutex::new(conn),
    });

    println!("Server running at http://localhost:8080");

    HttpServer::new(move || {
        let cors = Cors::permissive();
        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .route("/api/register", web::post().to(register))
            .route("/api/login", web::post().to(login))
            .route("/api/tweets", web::get().to(get_tweets))
            .route("/api/tweets", web::post().to(create_tweet))
            .route("/api/tweets/search", web::get().to(search_tweets))
            .route("/api/tweets/{id}/like", web::post().to(like_tweet))
            .route("/api/tweets/{id}", web::delete().to(delete_tweet))
            .route("/api/upload", web::post().to(upload_image))
            .route("/uploads/{filename}", web::get().to(serve_upload))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
