use axum::extract::{Extension, Path, State};
use axum::Json;
use sqlx::SqlitePool;

use crate::errors::AppError;
use crate::models::{Claims, CreatePostRequest, Post, PostResponse};

async fn enrich_post(pool: &SqlitePool, post: Post, user_id: &str) -> Result<PostResponse, AppError> {
    let username: (String,) =
        sqlx::query_as("SELECT username FROM users WHERE id = ?1")
            .bind(&post.user_id)
            .fetch_one(pool)
            .await?;

    let likes_count: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM likes WHERE post_id = ?1")
            .bind(&post.id)
            .fetch_one(pool)
            .await?;

    let liked: Option<(String,)> =
        sqlx::query_as("SELECT user_id FROM likes WHERE post_id = ?1 AND user_id = ?2")
            .bind(&post.id)
            .bind(user_id)
            .fetch_optional(pool)
            .await?;

    Ok(PostResponse {
        id: post.id,
        user_id: post.user_id,
        username: username.0,
        content: post.content,
        likes_count: likes_count.0,
        liked_by_me: liked.is_some(),
        created_at: post.created_at,
    })
}

pub async fn get_timeline(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<Vec<PostResponse>>, AppError> {
    let posts: Vec<Post> = sqlx::query_as(
        "SELECT p.id, p.user_id, p.content, p.created_at FROM posts p WHERE p.user_id IN (SELECT following_id FROM follows WHERE follower_id = ?1) OR p.user_id = ?1 ORDER BY p.created_at DESC LIMIT 50",
    )
    .bind(&claims.sub)
    .fetch_all(&pool)
    .await?;

    let mut result = Vec::new();
    for post in posts {
        result.push(enrich_post(&pool, post, &claims.sub).await?);
    }

    Ok(Json(result))
}

pub async fn create_post(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Json(body): Json<CreatePostRequest>,
) -> Result<Json<PostResponse>, AppError> {
    if body.content.is_empty() {
        return Err(AppError::BadRequest("Post content cannot be empty".to_string()));
    }

    if body.content.len() > 280 {
        return Err(AppError::BadRequest(
            "Post content cannot exceed 280 characters".to_string(),
        ));
    }

    let id = uuid::Uuid::new_v4().to_string();
    let created_at = chrono::Utc::now().to_rfc3339();

    sqlx::query("INSERT INTO posts (id, user_id, content, created_at) VALUES (?1, ?2, ?3, ?4)")
        .bind(&id)
        .bind(&claims.sub)
        .bind(&body.content)
        .bind(&created_at)
        .execute(&pool)
        .await?;

    let post = Post {
        id,
        user_id: claims.sub.clone(),
        content: body.content,
        created_at,
    };

    let response = enrich_post(&pool, post, &claims.sub).await?;
    Ok(Json(response))
}

pub async fn get_post(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<PostResponse>, AppError> {
    let post: Post =
        sqlx::query_as("SELECT id, user_id, content, created_at FROM posts WHERE id = ?1")
            .bind(&id)
            .fetch_optional(&pool)
            .await?
            .ok_or_else(|| AppError::NotFound("Post not found".to_string()))?;

    let response = enrich_post(&pool, post, &claims.sub).await?;
    Ok(Json(response))
}

pub async fn delete_post(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let post: Post =
        sqlx::query_as("SELECT id, user_id, content, created_at FROM posts WHERE id = ?1")
            .bind(&id)
            .fetch_optional(&pool)
            .await?
            .ok_or_else(|| AppError::NotFound("Post not found".to_string()))?;

    if post.user_id != claims.sub {
        return Err(AppError::Unauthorized("Cannot delete another user's post".to_string()));
    }

    sqlx::query("DELETE FROM likes WHERE post_id = ?1")
        .bind(&id)
        .execute(&pool)
        .await?;

    sqlx::query("DELETE FROM posts WHERE id = ?1")
        .bind(&id)
        .execute(&pool)
        .await?;

    Ok(Json(serde_json::json!({ "message": "Post deleted" })))
}

pub async fn like_post(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _post: Post =
        sqlx::query_as("SELECT id, user_id, content, created_at FROM posts WHERE id = ?1")
            .bind(&id)
            .fetch_optional(&pool)
            .await?
            .ok_or_else(|| AppError::NotFound("Post not found".to_string()))?;

    let existing: Option<(String,)> =
        sqlx::query_as("SELECT user_id FROM likes WHERE user_id = ?1 AND post_id = ?2")
            .bind(&claims.sub)
            .bind(&id)
            .fetch_optional(&pool)
            .await?;

    if existing.is_some() {
        return Err(AppError::Conflict("Already liked this post".to_string()));
    }

    let created_at = chrono::Utc::now().to_rfc3339();

    sqlx::query("INSERT INTO likes (user_id, post_id, created_at) VALUES (?1, ?2, ?3)")
        .bind(&claims.sub)
        .bind(&id)
        .bind(&created_at)
        .execute(&pool)
        .await?;

    Ok(Json(serde_json::json!({ "message": "Post liked" })))
}

pub async fn unlike_post(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let result = sqlx::query("DELETE FROM likes WHERE user_id = ?1 AND post_id = ?2")
        .bind(&claims.sub)
        .bind(&id)
        .execute(&pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound("Like not found".to_string()));
    }

    Ok(Json(serde_json::json!({ "message": "Post unliked" })))
}

pub async fn get_user_posts(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
    Path(id): Path<String>,
) -> Result<Json<Vec<PostResponse>>, AppError> {
    let posts: Vec<Post> = sqlx::query_as(
        "SELECT id, user_id, content, created_at FROM posts WHERE user_id = ?1 ORDER BY created_at DESC",
    )
    .bind(&id)
    .fetch_all(&pool)
    .await?;

    let mut result = Vec::new();
    for post in posts {
        result.push(enrich_post(&pool, post, &claims.sub).await?);
    }

    Ok(Json(result))
}
