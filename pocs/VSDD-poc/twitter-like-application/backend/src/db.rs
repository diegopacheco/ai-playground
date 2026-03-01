use rusqlite::Connection;
use std::sync::Mutex;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::json;

pub struct AppState {
    pub db: Mutex<Connection>,
    pub sessions: Mutex<HashMap<String, i64>>,
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
}

pub fn init_db(conn: &Connection) {
    conn.execute_batch("
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            bio TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_id INTEGER NOT NULL REFERENCES users(id),
            content TEXT NOT NULL,
            image_url TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS likes (
            user_id INTEGER NOT NULL REFERENCES users(id),
            post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, post_id)
        );
        CREATE TABLE IF NOT EXISTS follows (
            follower_id INTEGER NOT NULL REFERENCES users(id),
            followee_id INTEGER NOT NULL REFERENCES users(id),
            created_at INTEGER NOT NULL,
            PRIMARY KEY (follower_id, followee_id),
            CHECK (follower_id != followee_id)
        );
    ").expect("Failed to init db");
}

pub fn seed_admin(conn: &Connection) {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE username = 'admin'",
        [],
        |row| row.get(0),
    ).unwrap_or(0);
    if count == 0 {
        let hash = bcrypt::hash("admin", 12).expect("Failed to hash password");
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, bio, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params!["admin", hash, "Admin", "", now()],
        ).expect("Failed to seed admin");
    }
}

pub fn register_user(state: &AppState, username: &str, password: &str, display_name: &str) -> Result<serde_json::Value, (u16, String)> {
    let hash = bcrypt::hash(password, 12).map_err(|e| (500, e.to_string()))?;
    let ts = now();
    let conn = state.db.lock().unwrap();
    let result = conn.execute(
        "INSERT INTO users (username, password_hash, display_name, bio, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![username, hash, display_name, "", ts],
    );
    match result {
        Err(e) if e.to_string().contains("UNIQUE") => return Err((409, "Username already taken".to_string())),
        Err(e) => return Err((500, e.to_string())),
        Ok(_) => {}
    }
    let id = conn.last_insert_rowid();
    let session_id = uuid::Uuid::new_v4().to_string();
    drop(conn);
    state.sessions.lock().unwrap().insert(session_id.clone(), id);
    Ok(json!({
        "user": {"id": id, "username": username, "display_name": display_name, "bio": "", "created_at": ts},
        "session_id": session_id
    }))
}

pub fn login_user(state: &AppState, username: &str, password: &str) -> Result<(serde_json::Value, String), (u16, String)> {
    let conn = state.db.lock().unwrap();
    let result = conn.query_row(
        "SELECT id, username, password_hash, display_name, bio, created_at FROM users WHERE username = ?1",
        [username],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
            ))
        },
    );
    match result {
        Ok((id, uname, hash, display_name, bio, created_at)) => {
            if !bcrypt::verify(password, &hash).unwrap_or(false) {
                return Err((401, "Invalid credentials".to_string()));
            }
            let session_id = uuid::Uuid::new_v4().to_string();
            drop(conn);
            state.sessions.lock().unwrap().insert(session_id.clone(), id);
            Ok((json!({
                "id": id, "username": uname, "display_name": display_name,
                "bio": bio, "created_at": created_at
            }), session_id))
        }
        Err(_) => Err((401, "Invalid credentials".to_string())),
    }
}

pub fn get_user_by_id(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    conn.query_row(
        "SELECT id, username, display_name, bio, created_at FROM users WHERE id = ?1",
        [user_id],
        |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "username": row.get::<_, String>(1)?,
                "display_name": row.get::<_, String>(2)?,
                "bio": row.get::<_, String>(3)?,
                "created_at": row.get::<_, i64>(4)?
            }))
        },
    ).map_err(|_| (404, "User not found".to_string()))
}

pub fn create_post(conn: &Connection, author_id: i64, content: &str, image_url: &str) -> Result<serde_json::Value, (u16, String)> {
    let ts = now();
    conn.execute(
        "INSERT INTO posts (author_id, content, image_url, created_at) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![author_id, content, image_url, ts],
    ).map_err(|e| (500, e.to_string()))?;
    let id = conn.last_insert_rowid();
    let (username, display_name) = conn.query_row(
        "SELECT username, display_name FROM users WHERE id = ?1",
        [author_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    ).map_err(|_| (404, "User not found".to_string()))?;
    Ok(json!({
        "id": id, "author_id": author_id, "author_username": username,
        "author_display_name": display_name, "content": content,
        "image_url": image_url, "like_count": 0, "liked_by_me": false,
        "created_at": ts
    }))
}

pub fn get_post(conn: &Connection, post_id: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    conn.query_row(
        "SELECT p.id, p.author_id, u.username, u.display_name, p.content, p.image_url, p.created_at,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p JOIN users u ON p.author_id = u.id WHERE p.id = ?1",
        [post_id],
        |row| {
            let pid: i64 = row.get(0)?;
            let aid: i64 = row.get(1)?;
            let liked = if let Some(uid) = current_user_id {
                conn.query_row(
                    "SELECT COUNT(*) FROM likes WHERE user_id = ?1 AND post_id = ?2",
                    rusqlite::params![uid, pid],
                    |r| r.get::<_, i64>(0),
                ).unwrap_or(0) > 0
            } else {
                false
            };
            Ok(json!({
                "id": pid, "author_id": aid,
                "author_username": row.get::<_, String>(2)?,
                "author_display_name": row.get::<_, String>(3)?,
                "content": row.get::<_, String>(4)?,
                "image_url": row.get::<_, String>(5)?,
                "created_at": row.get::<_, i64>(6)?,
                "like_count": row.get::<_, i64>(7)?,
                "liked_by_me": liked
            }))
        },
    ).map_err(|_| (404, "Post not found".to_string()))
}

pub fn delete_post(conn: &Connection, post_id: i64, user_id: i64) -> Result<String, (u16, String)> {
    let (author_id, image_url): (i64, String) = conn.query_row(
        "SELECT author_id, image_url FROM posts WHERE id = ?1",
        [post_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).map_err(|_| (404, "Post not found".to_string()))?;
    if author_id != user_id {
        return Err((403, "Not the author".to_string()));
    }
    conn.execute("DELETE FROM posts WHERE id = ?1", [post_id])
        .map_err(|e| (500, e.to_string()))?;
    Ok(image_url)
}

pub fn list_posts(conn: &Connection, page: i64, limit: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    let offset = crate::validation::pagination_offset(page, limit);
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM posts", [], |row| row.get(0)).unwrap_or(0);
    let mut stmt = conn.prepare(
        "SELECT p.id, p.author_id, u.username, u.display_name, p.content, p.image_url, p.created_at,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p JOIN users u ON p.author_id = u.id
         ORDER BY p.created_at DESC, p.id DESC LIMIT ?1 OFFSET ?2"
    ).map_err(|e| (500, e.to_string()))?;
    let posts: Vec<serde_json::Value> = stmt.query_map(rusqlite::params![limit, offset], |row| {
        let pid: i64 = row.get(0)?;
        let liked = if let Some(uid) = current_user_id {
            conn.query_row(
                "SELECT COUNT(*) FROM likes WHERE user_id = ?1 AND post_id = ?2",
                rusqlite::params![uid, pid],
                |r| r.get::<_, i64>(0),
            ).unwrap_or(0) > 0
        } else {
            false
        };
        Ok(json!({
            "id": pid, "author_id": row.get::<_, i64>(1)?,
            "author_username": row.get::<_, String>(2)?,
            "author_display_name": row.get::<_, String>(3)?,
            "content": row.get::<_, String>(4)?,
            "image_url": row.get::<_, String>(5)?,
            "created_at": row.get::<_, i64>(6)?,
            "like_count": row.get::<_, i64>(7)?,
            "liked_by_me": liked
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"posts": posts, "total": total}))
}

pub fn like_post(conn: &Connection, user_id: i64, post_id: i64) -> Result<serde_json::Value, (u16, String)> {
    let exists: i64 = conn.query_row("SELECT COUNT(*) FROM posts WHERE id = ?1", [post_id], |row| row.get(0)).unwrap_or(0);
    if exists == 0 {
        return Err((404, "Post not found".to_string()));
    }
    conn.execute(
        "INSERT OR IGNORE INTO likes (user_id, post_id, created_at) VALUES (?1, ?2, ?3)",
        rusqlite::params![user_id, post_id, now()],
    ).map_err(|e| (500, e.to_string()))?;
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM likes WHERE post_id = ?1", [post_id], |row| row.get(0)).unwrap_or(0);
    Ok(json!({"like_count": count}))
}

pub fn unlike_post(conn: &Connection, user_id: i64, post_id: i64) -> Result<serde_json::Value, (u16, String)> {
    let exists: i64 = conn.query_row("SELECT COUNT(*) FROM posts WHERE id = ?1", [post_id], |row| row.get(0)).unwrap_or(0);
    if exists == 0 {
        return Err((404, "Post not found".to_string()));
    }
    conn.execute(
        "DELETE FROM likes WHERE user_id = ?1 AND post_id = ?2",
        rusqlite::params![user_id, post_id],
    ).map_err(|e| (500, e.to_string()))?;
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM likes WHERE post_id = ?1", [post_id], |row| row.get(0)).unwrap_or(0);
    Ok(json!({"like_count": count}))
}

pub fn follow_user(conn: &Connection, follower_id: i64, followee_id: i64) -> Result<(), (u16, String)> {
    if follower_id == followee_id {
        return Err((400, "Cannot follow yourself".to_string()));
    }
    let exists: i64 = conn.query_row("SELECT COUNT(*) FROM users WHERE id = ?1", [followee_id], |row| row.get(0)).unwrap_or(0);
    if exists == 0 {
        return Err((404, "User not found".to_string()));
    }
    conn.execute(
        "INSERT OR IGNORE INTO follows (follower_id, followee_id, created_at) VALUES (?1, ?2, ?3)",
        rusqlite::params![follower_id, followee_id, now()],
    ).map_err(|e| (500, e.to_string()))?;
    Ok(())
}

pub fn unfollow_user(conn: &Connection, follower_id: i64, followee_id: i64) -> Result<(), (u16, String)> {
    let exists: i64 = conn.query_row("SELECT COUNT(*) FROM users WHERE id = ?1", [followee_id], |row| row.get(0)).unwrap_or(0);
    if exists == 0 {
        return Err((404, "User not found".to_string()));
    }
    conn.execute(
        "DELETE FROM follows WHERE follower_id = ?1 AND followee_id = ?2",
        rusqlite::params![follower_id, followee_id],
    ).map_err(|e| (500, e.to_string()))?;
    Ok(())
}

pub fn get_followers(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    let exists: i64 = conn.query_row("SELECT COUNT(*) FROM users WHERE id = ?1", [user_id], |row| row.get(0)).unwrap_or(0);
    if exists == 0 {
        return Err((404, "User not found".to_string()));
    }
    let mut stmt = conn.prepare(
        "SELECT u.id, u.username, u.display_name, u.bio, u.created_at
         FROM follows f JOIN users u ON f.follower_id = u.id WHERE f.followee_id = ?1"
    ).map_err(|e| (500, e.to_string()))?;
    let users: Vec<serde_json::Value> = stmt.query_map([user_id], |row| {
        Ok(json!({
            "id": row.get::<_, i64>(0)?,
            "username": row.get::<_, String>(1)?,
            "display_name": row.get::<_, String>(2)?,
            "bio": row.get::<_, String>(3)?,
            "created_at": row.get::<_, i64>(4)?
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"users": users}))
}

pub fn get_following(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    let exists: i64 = conn.query_row("SELECT COUNT(*) FROM users WHERE id = ?1", [user_id], |row| row.get(0)).unwrap_or(0);
    if exists == 0 {
        return Err((404, "User not found".to_string()));
    }
    let mut stmt = conn.prepare(
        "SELECT u.id, u.username, u.display_name, u.bio, u.created_at
         FROM follows f JOIN users u ON f.followee_id = u.id WHERE f.follower_id = ?1"
    ).map_err(|e| (500, e.to_string()))?;
    let users: Vec<serde_json::Value> = stmt.query_map([user_id], |row| {
        Ok(json!({
            "id": row.get::<_, i64>(0)?,
            "username": row.get::<_, String>(1)?,
            "display_name": row.get::<_, String>(2)?,
            "bio": row.get::<_, String>(3)?,
            "created_at": row.get::<_, i64>(4)?
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"users": users}))
}

pub fn get_timeline(conn: &Connection, user_id: i64, page: i64, limit: i64) -> Result<serde_json::Value, (u16, String)> {
    let offset = crate::validation::pagination_offset(page, limit);
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM posts p WHERE p.author_id = ?1 OR p.author_id IN (SELECT followee_id FROM follows WHERE follower_id = ?1)",
        [user_id],
        |row| row.get(0),
    ).unwrap_or(0);
    let mut stmt = conn.prepare(
        "SELECT p.id, p.author_id, u.username, u.display_name, p.content, p.image_url, p.created_at,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p JOIN users u ON p.author_id = u.id
         WHERE p.author_id = ?1 OR p.author_id IN (SELECT followee_id FROM follows WHERE follower_id = ?1)
         ORDER BY p.created_at DESC, p.id DESC LIMIT ?2 OFFSET ?3"
    ).map_err(|e| (500, e.to_string()))?;
    let posts: Vec<serde_json::Value> = stmt.query_map(rusqlite::params![user_id, limit, offset], |row| {
        let pid: i64 = row.get(0)?;
        let liked = conn.query_row(
            "SELECT COUNT(*) FROM likes WHERE user_id = ?1 AND post_id = ?2",
            rusqlite::params![user_id, pid],
            |r| r.get::<_, i64>(0),
        ).unwrap_or(0) > 0;
        Ok(json!({
            "id": pid, "author_id": row.get::<_, i64>(1)?,
            "author_username": row.get::<_, String>(2)?,
            "author_display_name": row.get::<_, String>(3)?,
            "content": row.get::<_, String>(4)?,
            "image_url": row.get::<_, String>(5)?,
            "created_at": row.get::<_, i64>(6)?,
            "like_count": row.get::<_, i64>(7)?,
            "liked_by_me": liked
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"posts": posts, "total": total}))
}

pub fn get_user_profile(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    let user = conn.query_row(
        "SELECT id, username, display_name, bio, created_at FROM users WHERE id = ?1",
        [user_id],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?, row.get::<_, i64>(4)?)),
    ).map_err(|_| (404, "User not found".to_string()))?;
    let post_count: i64 = conn.query_row("SELECT COUNT(*) FROM posts WHERE author_id = ?1", [user_id], |row| row.get(0)).unwrap_or(0);
    let follower_count: i64 = conn.query_row("SELECT COUNT(*) FROM follows WHERE followee_id = ?1", [user_id], |row| row.get(0)).unwrap_or(0);
    let following_count: i64 = conn.query_row("SELECT COUNT(*) FROM follows WHERE follower_id = ?1", [user_id], |row| row.get(0)).unwrap_or(0);
    Ok(json!({
        "id": user.0, "username": user.1, "display_name": user.2,
        "bio": user.3, "created_at": user.4,
        "post_count": post_count, "follower_count": follower_count, "following_count": following_count
    }))
}

pub fn update_user_profile(conn: &Connection, user_id: i64, display_name: &str, bio: &str) -> Result<serde_json::Value, (u16, String)> {
    conn.execute(
        "UPDATE users SET display_name = ?1, bio = ?2 WHERE id = ?3",
        rusqlite::params![display_name, bio, user_id],
    ).map_err(|e| (500, e.to_string()))?;
    get_user_by_id(conn, user_id)
}

pub fn get_user_posts(conn: &Connection, user_id: i64, page: i64, limit: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    let exists: i64 = conn.query_row("SELECT COUNT(*) FROM users WHERE id = ?1", [user_id], |row| row.get(0)).unwrap_or(0);
    if exists == 0 {
        return Err((404, "User not found".to_string()));
    }
    let offset = crate::validation::pagination_offset(page, limit);
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM posts WHERE author_id = ?1", [user_id], |row| row.get(0)).unwrap_or(0);
    let mut stmt = conn.prepare(
        "SELECT p.id, p.author_id, u.username, u.display_name, p.content, p.image_url, p.created_at,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p JOIN users u ON p.author_id = u.id WHERE p.author_id = ?1
         ORDER BY p.created_at DESC, p.id DESC LIMIT ?2 OFFSET ?3"
    ).map_err(|e| (500, e.to_string()))?;
    let posts: Vec<serde_json::Value> = stmt.query_map(rusqlite::params![user_id, limit, offset], |row| {
        let pid: i64 = row.get(0)?;
        let liked = if let Some(uid) = current_user_id {
            conn.query_row(
                "SELECT COUNT(*) FROM likes WHERE user_id = ?1 AND post_id = ?2",
                rusqlite::params![uid, pid],
                |r| r.get::<_, i64>(0),
            ).unwrap_or(0) > 0
        } else {
            false
        };
        Ok(json!({
            "id": pid, "author_id": row.get::<_, i64>(1)?,
            "author_username": row.get::<_, String>(2)?,
            "author_display_name": row.get::<_, String>(3)?,
            "content": row.get::<_, String>(4)?,
            "image_url": row.get::<_, String>(5)?,
            "created_at": row.get::<_, i64>(6)?,
            "like_count": row.get::<_, i64>(7)?,
            "liked_by_me": liked
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"posts": posts, "total": total}))
}

pub fn search_posts(conn: &Connection, query: &str, page: i64, limit: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    let escaped = crate::validation::escape_search_term(query);
    let pattern = format!("%{}%", escaped);
    let offset = crate::validation::pagination_offset(page, limit);
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM posts WHERE content LIKE ?1 ESCAPE '\\'",
        [&pattern],
        |row| row.get(0),
    ).unwrap_or(0);
    let mut stmt = conn.prepare(
        "SELECT p.id, p.author_id, u.username, u.display_name, p.content, p.image_url, p.created_at,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p JOIN users u ON p.author_id = u.id
         WHERE p.content LIKE ?1 ESCAPE '\\'
         ORDER BY p.created_at DESC, p.id DESC LIMIT ?2 OFFSET ?3"
    ).map_err(|e| (500, e.to_string()))?;
    let posts: Vec<serde_json::Value> = stmt.query_map(rusqlite::params![pattern, limit, offset], |row| {
        let pid: i64 = row.get(0)?;
        let liked = if let Some(uid) = current_user_id {
            conn.query_row(
                "SELECT COUNT(*) FROM likes WHERE user_id = ?1 AND post_id = ?2",
                rusqlite::params![uid, pid],
                |r| r.get::<_, i64>(0),
            ).unwrap_or(0) > 0
        } else {
            false
        };
        Ok(json!({
            "id": pid, "author_id": row.get::<_, i64>(1)?,
            "author_username": row.get::<_, String>(2)?,
            "author_display_name": row.get::<_, String>(3)?,
            "content": row.get::<_, String>(4)?,
            "image_url": row.get::<_, String>(5)?,
            "created_at": row.get::<_, i64>(6)?,
            "like_count": row.get::<_, i64>(7)?,
            "liked_by_me": liked
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"posts": posts, "total": total}))
}

pub fn search_users(conn: &Connection, query: &str, page: i64, limit: i64) -> Result<serde_json::Value, (u16, String)> {
    let escaped = crate::validation::escape_search_term(query);
    let pattern = format!("%{}%", escaped);
    let offset = crate::validation::pagination_offset(page, limit);
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE username LIKE ?1 ESCAPE '\\' OR display_name LIKE ?1 ESCAPE '\\'",
        [&pattern],
        |row| row.get(0),
    ).unwrap_or(0);
    let mut stmt = conn.prepare(
        "SELECT id, username, display_name, bio, created_at FROM users
         WHERE username LIKE ?1 ESCAPE '\\' OR display_name LIKE ?1 ESCAPE '\\'
         ORDER BY username ASC LIMIT ?2 OFFSET ?3"
    ).map_err(|e| (500, e.to_string()))?;
    let users: Vec<serde_json::Value> = stmt.query_map(rusqlite::params![pattern, limit, offset], |row| {
        Ok(json!({
            "id": row.get::<_, i64>(0)?,
            "username": row.get::<_, String>(1)?,
            "display_name": row.get::<_, String>(2)?,
            "bio": row.get::<_, String>(3)?,
            "created_at": row.get::<_, i64>(4)?
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"users": users, "total": total}))
}

pub fn get_hot_posts(conn: &Connection, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    let cutoff = now() - 86400;
    let mut stmt = conn.prepare(
        "SELECT p.id, p.author_id, u.username, u.display_name, p.content, p.image_url, p.created_at,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p JOIN users u ON p.author_id = u.id
         WHERE p.created_at >= ?1 AND (SELECT COUNT(*) FROM likes WHERE post_id = p.id) > 0
         ORDER BY like_count DESC, p.created_at DESC, p.id DESC LIMIT 10"
    ).map_err(|e| (500, e.to_string()))?;
    let posts: Vec<serde_json::Value> = stmt.query_map([cutoff], |row| {
        let pid: i64 = row.get(0)?;
        let liked = if let Some(uid) = current_user_id {
            conn.query_row(
                "SELECT COUNT(*) FROM likes WHERE user_id = ?1 AND post_id = ?2",
                rusqlite::params![uid, pid],
                |r| r.get::<_, i64>(0),
            ).unwrap_or(0) > 0
        } else {
            false
        };
        Ok(json!({
            "id": pid, "author_id": row.get::<_, i64>(1)?,
            "author_username": row.get::<_, String>(2)?,
            "author_display_name": row.get::<_, String>(3)?,
            "content": row.get::<_, String>(4)?,
            "image_url": row.get::<_, String>(5)?,
            "created_at": row.get::<_, i64>(6)?,
            "like_count": row.get::<_, i64>(7)?,
            "liked_by_me": liked
        }))
    }).map_err(|e| (500, e.to_string()))?.filter_map(|r| r.ok()).collect();
    Ok(json!({"posts": posts}))
}
