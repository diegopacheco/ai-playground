use axum::extract::{Path, Query, State};
use axum::Json;
use sqlx::SqlitePool;

use crate::auth::AuthUser;
use crate::errors::AppError;
use crate::models::post::{CreatePostRequest, PaginationParams, PostResponse};

pub async fn create_post(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
    Json(req): Json<CreatePostRequest>,
) -> Result<Json<PostResponse>, AppError> {
    if req.content.is_empty() {
        return Err(AppError::BadRequest("Content cannot be empty".to_string()));
    }
    if req.content.len() > 280 {
        return Err(AppError::BadRequest("Content exceeds 280 characters".to_string()));
    }

    let result = sqlx::query(
        "INSERT INTO posts (user_id, content) VALUES (?, ?)"
    )
    .bind(auth.user_id)
    .bind(&req.content)
    .execute(&pool)
    .await?;

    let post_id = result.last_insert_rowid();

    let post = sqlx::query_as::<_, PostRow>(
        "SELECT p.id, p.user_id, p.content, p.created_at, u.username,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p
         INNER JOIN users u ON u.id = p.user_id
         WHERE p.id = ?"
    )
    .bind(post_id)
    .fetch_one(&pool)
    .await?;

    Ok(Json(post.into_response()))
}

pub async fn get_post(
    State(pool): State<SqlitePool>,
    Path(id): Path<i64>,
) -> Result<Json<PostResponse>, AppError> {
    let post = sqlx::query_as::<_, PostRow>(
        "SELECT p.id, p.user_id, p.content, p.created_at, u.username,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p
         INNER JOIN users u ON u.id = p.user_id
         WHERE p.id = ?"
    )
    .bind(id)
    .fetch_optional(&pool)
    .await?
    .ok_or_else(|| AppError::NotFound("Post not found".to_string()))?;

    Ok(Json(post.into_response()))
}

pub async fn delete_post(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, AppError> {
    let post = sqlx::query_as::<_, crate::models::post::Post>(
        "SELECT * FROM posts WHERE id = ?"
    )
    .bind(id)
    .fetch_optional(&pool)
    .await?
    .ok_or_else(|| AppError::NotFound("Post not found".to_string()))?;

    if post.user_id != auth.user_id {
        return Err(AppError::Unauthorized("You can only delete your own posts".to_string()));
    }

    sqlx::query("DELETE FROM likes WHERE post_id = ?")
        .bind(id)
        .execute(&pool)
        .await?;

    sqlx::query("DELETE FROM posts WHERE id = ?")
        .bind(id)
        .execute(&pool)
        .await?;

    Ok(Json(serde_json::json!({ "message": "Post deleted" })))
}

pub async fn get_all_posts(
    State(pool): State<SqlitePool>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<PostResponse>>, AppError> {
    let page = params.page.unwrap_or(1).max(1);
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = (page - 1) * limit;

    let posts = sqlx::query_as::<_, PostRow>(
        "SELECT p.id, p.user_id, p.content, p.created_at, u.username,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p
         INNER JOIN users u ON u.id = p.user_id
         ORDER BY p.created_at DESC
         LIMIT ? OFFSET ?"
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(&pool)
    .await?;

    let response: Vec<PostResponse> = posts.into_iter().map(|p| p.into_response()).collect();
    Ok(Json(response))
}

#[derive(sqlx::FromRow)]
struct PostRow {
    id: i64,
    user_id: i64,
    content: String,
    created_at: String,
    username: String,
    like_count: i32,
}

impl PostRow {
    fn into_response(self) -> PostResponse {
        PostResponse {
            id: self.id,
            user_id: self.user_id,
            content: self.content,
            created_at: self.created_at,
            username: self.username,
            like_count: self.like_count as i64,
        }
    }
}
