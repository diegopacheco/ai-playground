use axum::extract::{Extension, Path, State};
use axum::Json;
use sqlx::SqlitePool;

use crate::errors::AppError;
use crate::models::{Claims, ProfileResponse, User, UserResponse};

pub async fn get_user(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<ProfileResponse>, AppError> {
    let user: User =
        sqlx::query_as("SELECT id, username, email, password_hash, created_at FROM users WHERE id = ?1")
            .bind(&id)
            .fetch_optional(&pool)
            .await?
            .ok_or_else(|| AppError::NotFound("User not found".to_string()))?;

    let followers_count: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM follows WHERE following_id = ?1")
            .bind(&id)
            .fetch_one(&pool)
            .await?;

    let following_count: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM follows WHERE follower_id = ?1")
            .bind(&id)
            .fetch_one(&pool)
            .await?;

    let is_following: Option<(String,)> =
        sqlx::query_as("SELECT follower_id FROM follows WHERE follower_id = ?1 AND following_id = ?2")
            .bind(&claims.sub)
            .bind(&id)
            .fetch_optional(&pool)
            .await?;

    let user_response: UserResponse = user.into();

    Ok(Json(ProfileResponse {
        user: user_response,
        followers_count: followers_count.0,
        following_count: following_count.0,
        is_following: is_following.is_some(),
    }))
}

pub async fn get_followers(
    State(pool): State<SqlitePool>,
    Path(id): Path<String>,
) -> Result<Json<Vec<UserResponse>>, AppError> {
    let followers: Vec<User> = sqlx::query_as(
        "SELECT u.id, u.username, u.email, u.password_hash, u.created_at FROM users u INNER JOIN follows f ON u.id = f.follower_id WHERE f.following_id = ?1",
    )
    .bind(&id)
    .fetch_all(&pool)
    .await?;

    let result: Vec<UserResponse> = followers.into_iter().map(|u| u.into()).collect();
    Ok(Json(result))
}

pub async fn get_following(
    State(pool): State<SqlitePool>,
    Path(id): Path<String>,
) -> Result<Json<Vec<UserResponse>>, AppError> {
    let following: Vec<User> = sqlx::query_as(
        "SELECT u.id, u.username, u.email, u.password_hash, u.created_at FROM users u INNER JOIN follows f ON u.id = f.following_id WHERE f.follower_id = ?1",
    )
    .bind(&id)
    .fetch_all(&pool)
    .await?;

    let result: Vec<UserResponse> = following.into_iter().map(|u| u.into()).collect();
    Ok(Json(result))
}

pub async fn follow_user(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if claims.sub == id {
        return Err(AppError::BadRequest("Cannot follow yourself".to_string()));
    }

    let target: Option<User> =
        sqlx::query_as("SELECT id, username, email, password_hash, created_at FROM users WHERE id = ?1")
            .bind(&id)
            .fetch_optional(&pool)
            .await?;

    if target.is_none() {
        return Err(AppError::NotFound("User not found".to_string()));
    }

    let existing: Option<(String,)> =
        sqlx::query_as("SELECT follower_id FROM follows WHERE follower_id = ?1 AND following_id = ?2")
            .bind(&claims.sub)
            .bind(&id)
            .fetch_optional(&pool)
            .await?;

    if existing.is_some() {
        return Err(AppError::Conflict("Already following this user".to_string()));
    }

    let created_at = chrono::Utc::now().to_rfc3339();

    sqlx::query("INSERT INTO follows (follower_id, following_id, created_at) VALUES (?1, ?2, ?3)")
        .bind(&claims.sub)
        .bind(&id)
        .bind(&created_at)
        .execute(&pool)
        .await?;

    Ok(Json(serde_json::json!({ "message": "Followed successfully" })))
}

pub async fn unfollow_user(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let result =
        sqlx::query("DELETE FROM follows WHERE follower_id = ?1 AND following_id = ?2")
            .bind(&claims.sub)
            .bind(&id)
            .execute(&pool)
            .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound("Not following this user".to_string()));
    }

    Ok(Json(serde_json::json!({ "message": "Unfollowed successfully" })))
}
