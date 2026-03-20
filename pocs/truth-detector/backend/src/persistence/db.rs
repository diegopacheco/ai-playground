use crate::models::types::{Analysis, CommitResult, LeaderboardEntry, TrackedUser, WeeklyScore};
use rusqlite::{Connection, Result, params};

pub fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            github_user TEXT NOT NULL,
            created_at TEXT NOT NULL,
            total_score INTEGER,
            avg_score REAL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS commits (
            id TEXT PRIMARY KEY,
            analysis_id TEXT NOT NULL REFERENCES analyses(id),
            repo_name TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            commit_message TEXT NOT NULL,
            commit_date TEXT NOT NULL,
            classification TEXT NOT NULL,
            summary TEXT NOT NULL,
            score INTEGER NOT NULL,
            fallback INTEGER NOT NULL DEFAULT 0,
            raw_llm_output TEXT
        );
        CREATE TABLE IF NOT EXISTS weekly_scores (
            id TEXT PRIMARY KEY,
            github_user TEXT NOT NULL,
            week_start TEXT NOT NULL,
            week_end TEXT NOT NULL,
            total_score INTEGER NOT NULL,
            avg_score REAL NOT NULL,
            num_commits INTEGER NOT NULL,
            deep_count INTEGER NOT NULL,
            decent_count INTEGER NOT NULL,
            shallow_count INTEGER NOT NULL,
            UNIQUE(github_user, week_start)
        );
        CREATE TABLE IF NOT EXISTS tracked_users (
            github_user TEXT PRIMARY KEY,
            added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS github_cache (
            id TEXT PRIMARY KEY,
            github_user TEXT NOT NULL,
            payload TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            UNIQUE(github_user)
        );",
    )?;
    Ok(())
}

pub fn insert_analysis(conn: &Connection, id: &str, github_user: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO analyses (id, github_user, created_at, status) VALUES (?1, ?2, ?3, 'running')",
        params![id, github_user, now],
    )?;
    Ok(())
}

pub fn update_analysis_complete(
    conn: &Connection,
    id: &str,
    total_score: i32,
    avg_score: f64,
) -> Result<()> {
    conn.execute(
        "UPDATE analyses SET status = 'completed', total_score = ?1, avg_score = ?2 WHERE id = ?3",
        params![total_score, avg_score, id],
    )?;
    Ok(())
}

pub fn update_analysis_failed(conn: &Connection, id: &str) -> Result<()> {
    conn.execute(
        "UPDATE analyses SET status = 'failed' WHERE id = ?1",
        params![id],
    )?;
    Ok(())
}

pub fn get_analysis(conn: &Connection, id: &str) -> Result<Option<Analysis>> {
    let mut stmt = conn.prepare(
        "SELECT id, github_user, created_at, total_score, avg_score, status FROM analyses WHERE id = ?1",
    )?;
    let mut rows = stmt.query(params![id])?;
    let row = match rows.next()? {
        Some(r) => r,
        None => return Ok(None),
    };
    let analysis_id: String = row.get(0)?;
    let github_user: String = row.get(1)?;
    let created_at: String = row.get(2)?;
    let total_score: Option<i32> = row.get(3)?;
    let avg_score: Option<f64> = row.get(4)?;
    let status: String = row.get(5)?;

    let commits = get_commits_for_analysis(conn, &analysis_id)?;

    Ok(Some(Analysis {
        id: analysis_id,
        github_user,
        created_at,
        total_score,
        avg_score,
        status,
        commits,
    }))
}

pub fn get_latest_analysis(conn: &Connection, username: &str) -> Result<Option<Analysis>> {
    let mut stmt = conn.prepare(
        "SELECT id FROM analyses WHERE github_user = ?1 ORDER BY created_at DESC LIMIT 1",
    )?;
    let mut rows = stmt.query(params![username])?;
    match rows.next()? {
        Some(row) => {
            let id: String = row.get(0)?;
            get_analysis(conn, &id)
        }
        None => Ok(None),
    }
}

fn get_commits_for_analysis(conn: &Connection, analysis_id: &str) -> Result<Vec<CommitResult>> {
    let mut stmt = conn.prepare(
        "SELECT id, analysis_id, repo_name, commit_sha, commit_message, commit_date, classification, summary, score, fallback, raw_llm_output FROM commits WHERE analysis_id = ?1",
    )?;
    let rows = stmt.query_map(params![analysis_id], |row| {
        Ok(CommitResult {
            id: row.get(0)?,
            analysis_id: row.get(1)?,
            repo_name: row.get(2)?,
            commit_sha: row.get(3)?,
            commit_message: row.get(4)?,
            commit_date: row.get(5)?,
            classification: row.get(6)?,
            summary: row.get(7)?,
            score: row.get(8)?,
            fallback: row.get::<_, i32>(9)? != 0,
            raw_llm_output: row.get(10)?,
        })
    })?;
    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

pub fn insert_commit_result(conn: &Connection, c: &CommitResult) -> Result<()> {
    conn.execute(
        "INSERT INTO commits (id, analysis_id, repo_name, commit_sha, commit_message, commit_date, classification, summary, score, fallback, raw_llm_output) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![c.id, c.analysis_id, c.repo_name, c.commit_sha, c.commit_message, c.commit_date, c.classification, c.summary, c.score, c.fallback as i32, c.raw_llm_output],
    )?;
    Ok(())
}

pub fn insert_commit_results(conn: &Connection, commits: &[CommitResult]) -> Result<()> {
    for c in commits {
        insert_commit_result(conn, c)?;
    }
    Ok(())
}

pub fn upsert_weekly_score(conn: &Connection, ws: &WeeklyScore) -> Result<()> {
    conn.execute(
        "INSERT INTO weekly_scores (id, github_user, week_start, week_end, total_score, avg_score, num_commits, deep_count, decent_count, shallow_count) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) ON CONFLICT(github_user, week_start) DO UPDATE SET total_score = ?5, avg_score = ?6, num_commits = ?7, deep_count = ?8, decent_count = ?9, shallow_count = ?10",
        params![ws.id, ws.github_user, ws.week_start, ws.week_end, ws.total_score, ws.avg_score, ws.num_commits, ws.deep_count, ws.decent_count, ws.shallow_count],
    )?;
    Ok(())
}

pub fn get_weekly_scores(conn: &Connection, username: &str) -> Result<Vec<WeeklyScore>> {
    let mut stmt = conn.prepare(
        "SELECT id, github_user, week_start, week_end, total_score, avg_score, num_commits, deep_count, decent_count, shallow_count FROM weekly_scores WHERE github_user = ?1 ORDER BY week_start DESC",
    )?;
    let rows = stmt.query_map(params![username], |row| {
        Ok(WeeklyScore {
            id: row.get(0)?,
            github_user: row.get(1)?,
            week_start: row.get(2)?,
            week_end: row.get(3)?,
            total_score: row.get(4)?,
            avg_score: row.get(5)?,
            num_commits: row.get(6)?,
            deep_count: row.get(7)?,
            decent_count: row.get(8)?,
            shallow_count: row.get(9)?,
        })
    })?;
    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

pub fn get_leaderboard(conn: &Connection, week_start: &str) -> Result<Vec<LeaderboardEntry>> {
    let mut stmt = conn.prepare(
        "SELECT github_user, avg_score, total_score, num_commits, deep_count, decent_count, shallow_count FROM weekly_scores WHERE week_start = ?1 ORDER BY avg_score DESC",
    )?;
    let rows = stmt.query_map(params![week_start], |row| {
        Ok(LeaderboardEntry {
            rank: 0,
            github_user: row.get(0)?,
            avg_score: row.get(1)?,
            total_score: row.get(2)?,
            num_commits: row.get(3)?,
            deep_count: row.get(4)?,
            decent_count: row.get(5)?,
            shallow_count: row.get(6)?,
        })
    })?;
    let mut entries = Vec::new();
    for (i, row) in rows.enumerate() {
        let mut entry = row?;
        entry.rank = (i + 1) as i32;
        entries.push(entry);
    }
    Ok(entries)
}

pub fn insert_tracked_user(conn: &Connection, github_user: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR IGNORE INTO tracked_users (github_user, added_at) VALUES (?1, ?2)",
        params![github_user, now],
    )?;
    Ok(())
}

pub fn delete_tracked_user(conn: &Connection, github_user: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM tracked_users WHERE github_user = ?1",
        params![github_user],
    )?;
    Ok(())
}

pub fn get_tracked_users(conn: &Connection) -> Result<Vec<TrackedUser>> {
    let mut stmt = conn.prepare("SELECT github_user, added_at FROM tracked_users")?;
    let rows = stmt.query_map([], |row| {
        Ok(TrackedUser {
            github_user: row.get(0)?,
            added_at: row.get(1)?,
        })
    })?;
    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

pub fn get_cached_github(conn: &Connection, github_user: &str) -> Result<Option<String>> {
    let now = chrono::Utc::now().to_rfc3339();
    let mut stmt = conn.prepare(
        "SELECT payload FROM github_cache WHERE github_user = ?1 AND expires_at > ?2",
    )?;
    let mut rows = stmt.query(params![github_user, now])?;
    match rows.next()? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

pub fn upsert_github_cache(conn: &Connection, github_user: &str, payload: &str) -> Result<()> {
    let now = chrono::Utc::now();
    let fetched_at = now.to_rfc3339();
    let expires_at = (now + chrono::Duration::hours(1)).to_rfc3339();
    let id = uuid::Uuid::new_v4().to_string();
    conn.execute(
        "INSERT INTO github_cache (id, github_user, payload, fetched_at, expires_at) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(github_user) DO UPDATE SET payload = ?3, fetched_at = ?4, expires_at = ?5",
        params![id, github_user, payload, fetched_at, expires_at],
    )?;
    Ok(())
}
