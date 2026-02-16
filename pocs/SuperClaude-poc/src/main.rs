use actix_web::{web, App, HttpServer, HttpResponse};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use chrono::Utc;

struct AppState {
    db: Mutex<Connection>,
}

#[derive(Serialize, Deserialize)]
struct Tweet {
    id: i64,
    username: String,
    content: String,
    likes: i64,
    created_at: String,
}

#[derive(Deserialize)]
struct CreateTweet {
    username: String,
    content: String,
}

fn init_db(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            content TEXT NOT NULL,
            likes INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )"
    ).unwrap();
}

async fn index() -> HttpResponse {
    HttpResponse::Ok().content_type("text/html").body(HTML)
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
    if body.content.is_empty() || body.content.len() > 280 {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": "Tweet must be 1-280 characters"}));
    }
    let username = if body.username.is_empty() {
        "Anonymous".to_string()
    } else {
        body.username.clone()
    };
    let now = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let conn = data.db.lock().unwrap();
    conn.execute(
        "INSERT INTO tweets (username, content, likes, created_at) VALUES (?1, ?2, 0, ?3)",
        rusqlite::params![username, body.content, now],
    ).unwrap();
    HttpResponse::Created().json(serde_json::json!({"status": "ok"}))
}

async fn like_tweet(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let rows = conn
        .execute("UPDATE tweets SET likes = likes + 1 WHERE id = ?1", rusqlite::params![id])
        .unwrap();
    if rows == 0 {
        return HttpResponse::NotFound().json(serde_json::json!({"error": "Tweet not found"}));
    }
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

async fn delete_tweet(data: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let rows = conn
        .execute("DELETE FROM tweets WHERE id = ?1", rusqlite::params![id])
        .unwrap();
    if rows == 0 {
        return HttpResponse::NotFound().json(serde_json::json!({"error": "Tweet not found"}));
    }
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let conn = Connection::open("tweets.db").unwrap();
    init_db(&conn);
    let data = web::Data::new(AppState {
        db: Mutex::new(conn),
    });
    println!("Server running at http://localhost:8080");
    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .route("/", web::get().to(index))
            .route("/api/tweets", web::get().to(get_tweets))
            .route("/api/tweets", web::post().to(create_tweet))
            .route("/api/tweets/{id}/like", web::post().to(like_tweet))
            .route("/api/tweets/{id}", web::delete().to(delete_tweet))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

const HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rustbird</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #15202b; color: #e1e8ed; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.container { max-width: 600px; margin: 0 auto; border-left: 1px solid #38444d; border-right: 1px solid #38444d; min-height: 100vh; }
.header { padding: 16px 20px; border-bottom: 1px solid #38444d; font-size: 20px; font-weight: 700; background: rgba(21, 32, 43, 0.95); position: sticky; top: 0; z-index: 10; }
.compose { padding: 16px 20px; border-bottom: 1px solid #38444d; }
.compose input, .compose textarea { width: 100%; background: #192734; border: 1px solid #38444d; color: #e1e8ed; border-radius: 8px; padding: 12px; font-size: 15px; margin-bottom: 10px; resize: none; font-family: inherit; }
.compose input:focus, .compose textarea:focus { outline: none; border-color: #1da1f2; }
.compose textarea { height: 80px; }
.compose-footer { display: flex; justify-content: space-between; align-items: center; }
.char-count { font-size: 13px; color: #8899a6; }
.char-count.warn { color: #e0245e; }
.btn { background: #1da1f2; color: white; border: none; border-radius: 20px; padding: 8px 20px; font-size: 15px; font-weight: 700; cursor: pointer; }
.btn:hover { background: #1a91da; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.tweet { padding: 16px 20px; border-bottom: 1px solid #38444d; transition: background 0.2s; }
.tweet:hover { background: #192734; }
.tweet-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.tweet-user { font-weight: 700; font-size: 15px; }
.tweet-time { font-size: 13px; color: #8899a6; }
.tweet-content { font-size: 15px; line-height: 1.4; margin-bottom: 12px; word-wrap: break-word; }
.tweet-actions { display: flex; gap: 24px; }
.tweet-action { display: flex; align-items: center; gap: 6px; color: #8899a6; cursor: pointer; font-size: 13px; background: none; border: none; padding: 4px 8px; border-radius: 20px; transition: all 0.2s; }
.tweet-action.like:hover { color: #e0245e; background: rgba(224, 36, 94, 0.1); }
.tweet-action.like.liked { color: #e0245e; }
.tweet-action.delete:hover { color: #e0245e; background: rgba(224, 36, 94, 0.1); }
.avatar { width: 40px; height: 40px; border-radius: 50%; background: #1da1f2; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; margin-right: 12px; flex-shrink: 0; }
.tweet-row { display: flex; }
.tweet-body { flex: 1; min-width: 0; }
.empty { text-align: center; padding: 40px 20px; color: #8899a6; font-size: 15px; }
</style>
</head>
<body>
<div class="container">
    <div class="header">Rustbird</div>
    <div class="compose">
        <input type="text" id="username" placeholder="Your name" maxlength="30">
        <textarea id="content" placeholder="What's happening?" maxlength="280"></textarea>
        <div class="compose-footer">
            <span class="char-count" id="charCount">280</span>
            <button class="btn" id="tweetBtn" onclick="postTweet()">Tweet</button>
        </div>
    </div>
    <div id="timeline"></div>
</div>
<script>
const timeline = document.getElementById('timeline');
const contentEl = document.getElementById('content');
const charCount = document.getElementById('charCount');
const tweetBtn = document.getElementById('tweetBtn');
const usernameEl = document.getElementById('username');

contentEl.addEventListener('input', () => {
    const remaining = 280 - contentEl.value.length;
    charCount.textContent = remaining;
    charCount.className = 'char-count' + (remaining < 20 ? ' warn' : '');
    tweetBtn.disabled = contentEl.value.trim().length === 0;
});

async function loadTweets() {
    const res = await fetch('/api/tweets');
    const tweets = await res.json();
    if (tweets.length === 0) {
        timeline.innerHTML = '<div class="empty">No tweets yet. Be the first!</div>';
        return;
    }
    timeline.innerHTML = tweets.map(t => {
        const initial = t.username.charAt(0).toUpperCase();
        return `<div class="tweet" id="tweet-${t.id}">
            <div class="tweet-row">
                <div class="avatar">${initial}</div>
                <div class="tweet-body">
                    <div class="tweet-header">
                        <span class="tweet-user">${esc(t.username)}</span>
                        <span class="tweet-time">${t.created_at}</span>
                    </div>
                    <div class="tweet-content">${esc(t.content)}</div>
                    <div class="tweet-actions">
                        <button class="tweet-action like" onclick="likeTweet(${t.id})">&#9829; ${t.likes}</button>
                        <button class="tweet-action delete" onclick="deleteTweet(${t.id})">&#128465; Delete</button>
                    </div>
                </div>
            </div>
        </div>`;
    }).join('');
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

async function postTweet() {
    const content = contentEl.value.trim();
    if (!content) return;
    await fetch('/api/tweets', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ username: usernameEl.value.trim(), content })
    });
    contentEl.value = '';
    charCount.textContent = '280';
    charCount.className = 'char-count';
    tweetBtn.disabled = true;
    loadTweets();
}

async function likeTweet(id) {
    await fetch(`/api/tweets/${id}/like`, { method: 'POST' });
    loadTweets();
}

async function deleteTweet(id) {
    await fetch(`/api/tweets/${id}`, { method: 'DELETE' });
    loadTweets();
}

contentEl.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') postTweet();
});

loadTweets();
setInterval(loadTweets, 10000);
</script>
</body>
</html>"#;
