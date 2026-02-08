use sqlx::{sqlite::SqlitePool, Pool, Sqlite};
use crate::persistence::models::ProjectRecord;

pub async fn init_db() -> Pool<Sqlite> {
    let pool = SqlitePool::connect("sqlite:projects.db?mode=rwc")
        .await
        .expect("Failed to connect to database");

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            engine TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create projects table");

    pool
}

pub async fn create_project(pool: &Pool<Sqlite>, project: &ProjectRecord) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO projects (id, name, engine, status, created_at, completed_at, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&project.id)
    .bind(&project.name)
    .bind(&project.engine)
    .bind(&project.status)
    .bind(&project.created_at)
    .bind(&project.completed_at)
    .bind(&project.error)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_project_status(
    pool: &Pool<Sqlite>,
    id: &str,
    status: &str,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    let completed_at = if status == "done" || status == "error" {
        Some(chrono::Utc::now().to_rfc3339())
    } else {
        None
    };

    sqlx::query(
        r#"
        UPDATE projects SET status = ?, error = ?, completed_at = ? WHERE id = ?
        "#,
    )
    .bind(status)
    .bind(error)
    .bind(completed_at)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn get_project(pool: &Pool<Sqlite>, id: &str) -> Result<Option<ProjectRecord>, sqlx::Error> {
    let project = sqlx::query_as::<_, ProjectRecord>(
        r#"SELECT * FROM projects WHERE id = ?"#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(project)
}

pub async fn get_all_projects(pool: &Pool<Sqlite>) -> Result<Vec<ProjectRecord>, sqlx::Error> {
    let projects = sqlx::query_as::<_, ProjectRecord>(
        r#"SELECT * FROM projects ORDER BY created_at DESC"#,
    )
    .fetch_all(pool)
    .await?;
    Ok(projects)
}

pub async fn delete_project(pool: &Pool<Sqlite>, id: &str) -> Result<(), sqlx::Error> {
    sqlx::query(r#"DELETE FROM projects WHERE id = ?"#)
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}
