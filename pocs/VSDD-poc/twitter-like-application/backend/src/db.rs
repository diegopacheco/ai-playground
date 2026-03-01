use rusqlite::Connection;
use std::sync::Mutex;

pub struct AppState {
    pub db: Mutex<Connection>,
    pub sessions: Mutex<std::collections::HashMap<String, i64>>,
}

pub fn init_db(conn: &Connection) {
    todo!()
}

pub fn seed_admin(conn: &Connection) {
    todo!()
}

pub fn register_user(state: &AppState, username: &str, password: &str, display_name: &str) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn login_user(state: &AppState, username: &str, password: &str) -> Result<(serde_json::Value, String), (u16, String)> {
    todo!()
}

pub fn get_user_by_id(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn create_post(conn: &Connection, author_id: i64, content: &str, image_url: &str) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn get_post(conn: &Connection, post_id: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn delete_post(conn: &Connection, post_id: i64, user_id: i64) -> Result<String, (u16, String)> {
    todo!()
}

pub fn list_posts(conn: &Connection, page: i64, limit: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn like_post(conn: &Connection, user_id: i64, post_id: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn unlike_post(conn: &Connection, user_id: i64, post_id: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn follow_user(conn: &Connection, follower_id: i64, followee_id: i64) -> Result<(), (u16, String)> {
    todo!()
}

pub fn unfollow_user(conn: &Connection, follower_id: i64, followee_id: i64) -> Result<(), (u16, String)> {
    todo!()
}

pub fn get_followers(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn get_following(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn get_timeline(conn: &Connection, user_id: i64, page: i64, limit: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn get_user_profile(conn: &Connection, user_id: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn update_user_profile(conn: &Connection, user_id: i64, display_name: &str, bio: &str) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn get_user_posts(conn: &Connection, user_id: i64, page: i64, limit: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn search_posts(conn: &Connection, query: &str, page: i64, limit: i64, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn search_users(conn: &Connection, query: &str, page: i64, limit: i64) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}

pub fn get_hot_posts(conn: &Connection, current_user_id: Option<i64>) -> Result<serde_json::Value, (u16, String)> {
    todo!()
}
