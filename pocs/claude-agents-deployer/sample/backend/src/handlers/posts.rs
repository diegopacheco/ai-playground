use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::models::post::{CreatePost, Post, UpdatePost};

pub async fn create_post(
    State(pool): State<PgPool>,
    Json(payload): Json<CreatePost>,
) -> impl IntoResponse {
    let post = sqlx::query_as::<_, Post>(
        "INSERT INTO posts (id, title, content, author, created_at, updated_at) VALUES ($1, $2, $3, $4, NOW(), NOW()) RETURNING *",
    )
    .bind(Uuid::new_v4())
    .bind(&payload.title)
    .bind(&payload.content)
    .bind(&payload.author)
    .fetch_one(&pool)
    .await;

    match post {
        Ok(post) => (StatusCode::CREATED, Json(serde_json::json!(post))).into_response(),
        Err(e) => {
            tracing::error!("Failed to create post: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

pub async fn list_posts(State(pool): State<PgPool>) -> impl IntoResponse {
    let posts = sqlx::query_as::<_, Post>("SELECT * FROM posts ORDER BY created_at DESC")
        .fetch_all(&pool)
        .await;

    match posts {
        Ok(posts) => (StatusCode::OK, Json(serde_json::json!(posts))).into_response(),
        Err(e) => {
            tracing::error!("Failed to list posts: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

pub async fn get_post(
    State(pool): State<PgPool>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    let post = sqlx::query_as::<_, Post>("SELECT * FROM posts WHERE id = $1")
        .bind(id)
        .fetch_optional(&pool)
        .await;

    match post {
        Ok(Some(post)) => (StatusCode::OK, Json(serde_json::json!(post))).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Post not found"})),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("Failed to get post: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

pub async fn update_post(
    State(pool): State<PgPool>,
    Path(id): Path<Uuid>,
    Json(payload): Json<UpdatePost>,
) -> impl IntoResponse {
    let existing = sqlx::query_as::<_, Post>("SELECT * FROM posts WHERE id = $1")
        .bind(id)
        .fetch_optional(&pool)
        .await;

    match existing {
        Ok(Some(existing)) => {
            let title = payload.title.unwrap_or(existing.title);
            let content = payload.content.unwrap_or(existing.content);
            let author = payload.author.unwrap_or(existing.author);

            let post = sqlx::query_as::<_, Post>(
                "UPDATE posts SET title = $1, content = $2, author = $3, updated_at = NOW() WHERE id = $4 RETURNING *",
            )
            .bind(&title)
            .bind(&content)
            .bind(&author)
            .bind(id)
            .fetch_one(&pool)
            .await;

            match post {
                Ok(post) => (StatusCode::OK, Json(serde_json::json!(post))).into_response(),
                Err(e) => {
                    tracing::error!("Failed to update post: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": e.to_string()})),
                    )
                        .into_response()
                }
            }
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Post not found"})),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("Failed to find post for update: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

pub async fn delete_post(
    State(pool): State<PgPool>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    let result = sqlx::query("DELETE FROM posts WHERE id = $1")
        .bind(id)
        .execute(&pool)
        .await;

    match result {
        Ok(r) if r.rows_affected() > 0 => {
            (StatusCode::OK, Json(serde_json::json!({"deleted": true}))).into_response()
        }
        Ok(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Post not found"})),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("Failed to delete post: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}
