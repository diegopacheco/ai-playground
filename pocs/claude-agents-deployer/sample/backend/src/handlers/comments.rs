use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::models::comment::{Comment, CreateComment};

pub async fn create_comment(
    State(pool): State<PgPool>,
    Path(post_id): Path<Uuid>,
    Json(payload): Json<CreateComment>,
) -> impl IntoResponse {
    let comments_enabled =
        sqlx::query_scalar::<_, bool>("SELECT comments_enabled FROM settings WHERE id = 1")
            .fetch_optional(&pool)
            .await;

    match comments_enabled {
        Ok(Some(false)) => {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({"error": "Comments are disabled"})),
            )
                .into_response();
        }
        Ok(_) => {}
        Err(e) => {
            tracing::error!("Failed to fetch settings: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    }

    let post_exists = sqlx::query("SELECT id FROM posts WHERE id = $1")
        .bind(post_id)
        .fetch_optional(&pool)
        .await;

    match post_exists {
        Ok(Some(_)) => {
            let comment = sqlx::query_as::<_, Comment>(
                "INSERT INTO comments (id, content, author, post_id, created_at) VALUES ($1, $2, $3, $4, NOW()) RETURNING *",
            )
            .bind(Uuid::new_v4())
            .bind(&payload.content)
            .bind(&payload.author)
            .bind(post_id)
            .fetch_one(&pool)
            .await;

            match comment {
                Ok(comment) => {
                    (StatusCode::CREATED, Json(serde_json::json!(comment))).into_response()
                }
                Err(e) => {
                    tracing::error!("Failed to create comment: {}", e);
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
            tracing::error!("Failed to check post existence: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

pub async fn list_comments(
    State(pool): State<PgPool>,
    Path(post_id): Path<Uuid>,
) -> impl IntoResponse {
    let comments = sqlx::query_as::<_, Comment>(
        "SELECT * FROM comments WHERE post_id = $1 ORDER BY created_at DESC",
    )
    .bind(post_id)
    .fetch_all(&pool)
    .await;

    match comments {
        Ok(comments) => (StatusCode::OK, Json(serde_json::json!(comments))).into_response(),
        Err(e) => {
            tracing::error!("Failed to list comments: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

pub async fn delete_comment(State(pool): State<PgPool>, Path(id): Path<Uuid>) -> impl IntoResponse {
    let result = sqlx::query("DELETE FROM comments WHERE id = $1")
        .bind(id)
        .execute(&pool)
        .await;

    match result {
        Ok(r) if r.rows_affected() > 0 => {
            (StatusCode::OK, Json(serde_json::json!({"deleted": true}))).into_response()
        }
        Ok(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Comment not found"})),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("Failed to delete comment: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}
