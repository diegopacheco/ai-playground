use axum::extract::{Path, State};
use axum::Json;
use sqlx::SqlitePool;

use crate::auth::AuthUser;
use crate::errors::AppError;

pub async fn follow_user(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
    Path(user_id): Path<i64>,
) -> Result<Json<serde_json::Value>, AppError> {
    if auth.user_id == user_id {
        return Err(AppError::BadRequest("Cannot follow yourself".to_string()));
    }

    let user_exists = sqlx::query("SELECT id FROM users WHERE id = ?")
        .bind(user_id)
        .fetch_optional(&pool)
        .await?;

    if user_exists.is_none() {
        return Err(AppError::NotFound("User not found".to_string()));
    }

    let result = sqlx::query(
        "INSERT INTO follows (follower_id, following_id) VALUES (?, ?)"
    )
    .bind(auth.user_id)
    .bind(user_id)
    .execute(&pool)
    .await;

    match result {
        Ok(_) => Ok(Json(serde_json::json!({ "message": "User followed" }))),
        Err(sqlx::Error::Database(err)) if err.message().contains("UNIQUE") => {
            Err(AppError::Conflict("Already following this user".to_string()))
        }
        Err(err) => Err(AppError::from(err)),
    }
}

pub async fn unfollow_user(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
    Path(user_id): Path<i64>,
) -> Result<Json<serde_json::Value>, AppError> {
    let result = sqlx::query(
        "DELETE FROM follows WHERE follower_id = ? AND following_id = ?"
    )
    .bind(auth.user_id)
    .bind(user_id)
    .execute(&pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound("Follow relationship not found".to_string()));
    }

    Ok(Json(serde_json::json!({ "message": "User unfollowed" })))
}
