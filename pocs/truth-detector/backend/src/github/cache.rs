use crate::github::client;
use crate::github::types::FetchedCommit;
use crate::persistence::db;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::Mutex;

pub async fn get_or_fetch(
    db_conn: &Arc<Mutex<Connection>>,
    username: &str,
) -> Result<(Vec<FetchedCommit>, bool), Box<dyn std::error::Error + Send + Sync>> {
    {
        let conn = db_conn.lock().await;
        if let Ok(Some(payload)) = db::get_cached_github(&conn, username) {
            if let Ok(commits) = serde_json::from_str::<Vec<FetchedCommit>>(&payload) {
                return Ok((commits, true));
            }
        }
    }

    let commits = client::fetch_commits(username).await?;
    let payload = serde_json::to_string(&commits)?;

    {
        let conn = db_conn.lock().await;
        let _ = db::upsert_github_cache(&conn, username, &payload);
    }

    Ok((commits, false))
}
