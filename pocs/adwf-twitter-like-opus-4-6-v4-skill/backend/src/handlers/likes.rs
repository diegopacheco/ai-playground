use axum::extract::{Path, State};
use axum::Json;
use sqlx::SqlitePool;

use crate::auth::AuthUser;
use crate::errors::AppError;
use crate::models::like::LikeCountResponse;

pub async fn like_post(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
    Path(post_id): Path<i64>,
) -> Result<Json<serde_json::Value>, AppError> {
    let post_exists = sqlx::query("SELECT id FROM posts WHERE id = ?")
        .bind(post_id)
        .fetch_optional(&pool)
        .await?;

    if post_exists.is_none() {
        return Err(AppError::NotFound("Post not found".to_string()));
    }

    let result = sqlx::query(
        "INSERT INTO likes (user_id, post_id) VALUES (?, ?)"
    )
    .bind(auth.user_id)
    .bind(post_id)
    .execute(&pool)
    .await;

    match result {
        Ok(_) => Ok(Json(serde_json::json!({ "message": "Post liked" }))),
        Err(sqlx::Error::Database(err)) if err.message().contains("UNIQUE") => {
            Err(AppError::Conflict("Already liked this post".to_string()))
        }
        Err(err) => Err(AppError::from(err)),
    }
}

pub async fn unlike_post(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
    Path(post_id): Path<i64>,
) -> Result<Json<serde_json::Value>, AppError> {
    let result = sqlx::query(
        "DELETE FROM likes WHERE user_id = ? AND post_id = ?"
    )
    .bind(auth.user_id)
    .bind(post_id)
    .execute(&pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound("Like not found".to_string()));
    }

    Ok(Json(serde_json::json!({ "message": "Post unliked" })))
}

pub async fn get_like_count(
    State(pool): State<SqlitePool>,
    Path(post_id): Path<i64>,
) -> Result<Json<LikeCountResponse>, AppError> {
    let row: (i32,) = sqlx::query_as(
        "SELECT COUNT(*) FROM likes WHERE post_id = ?"
    )
    .bind(post_id)
    .fetch_one(&pool)
    .await?;

    Ok(Json(LikeCountResponse {
        post_id,
        count: row.0 as i64,
    }))
}
