use redis::{Commands, Connection};
use std::collections::HashSet;

pub fn connect() -> Connection {
    let client = redis::Client::open("redis://127.0.0.1:6379/").unwrap();
    client.get_connection().unwrap()
}

pub fn ensure_root(conn: &mut Connection) {
    let exists: bool = redis::cmd("EXISTS").arg("fs:meta:/").query(conn).unwrap();
    if !exists {
        set_meta(conn, "/", "dir", 0);
    }
}

pub fn set_meta(conn: &mut Connection, path: &str, file_type: &str, size: usize) {
    let _: () = redis::cmd("HSET")
        .arg(format!("fs:meta:{}", path))
        .arg("type")
        .arg(file_type)
        .arg("size")
        .arg(size.to_string())
        .query(conn)
        .unwrap();
}

pub fn set_data(conn: &mut Connection, path: &str, content: &str) {
    let _: () = conn.set(format!("fs:data:{}", path), content).unwrap();
}

pub fn get_data(conn: &mut Connection, path: &str) -> Option<String> {
    conn.get(format!("fs:data:{}", path)).unwrap()
}

pub fn del_entry(conn: &mut Connection, path: &str) {
    let _: () = conn.del(format!("fs:meta:{}", path)).unwrap();
    let _: () = conn.del(format!("fs:data:{}", path)).unwrap();
}

pub fn add_to_dir(conn: &mut Connection, dir: &str, name: &str) {
    let _: () = conn.sadd(format!("fs:dir:{}", dir), name).unwrap();
}

pub fn remove_from_dir(conn: &mut Connection, dir: &str, name: &str) {
    let _: () = conn.srem(format!("fs:dir:{}", dir), name).unwrap();
}

pub fn list_dir(conn: &mut Connection, path: &str) -> HashSet<String> {
    conn.smembers(format!("fs:dir:{}", path)).unwrap_or_default()
}
