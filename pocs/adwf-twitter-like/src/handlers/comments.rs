use axum::{
    extract::{Path, State},
    http::StatusCode,
    Extension,
    Json,
};
use std::sync::Arc;
use validator::Validate;

use crate::{
    error::{AppError, Result},
    middleware::Claims,
    models::{Comment, CommentResponse, CreateCommentRequest},
    state::AppState,
};

pub async fn create_comment(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
    Extension(claims): Extension<Claims>,
    Json(payload): Json<CreateCommentRequest>,
) -> Result<(StatusCode, Json<CommentResponse>)> {
    payload.validate().map_err(|e| AppError::BadRequest(e.to_string()))?;

    let tweet_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM tweets WHERE id = $1)"
    )
    .bind(tweet_id)
    .fetch_one(&state.db)
    .await?;

    if !tweet_exists {
        return Err(AppError::NotFound("Tweet not found".to_string()));
    }

    let comment = sqlx::query_as::<_, Comment>(
        r#"
        INSERT INTO comments (user_id, tweet_id, content)
        VALUES ($1, $2, $3)
        RETURNING id, user_id, tweet_id, content, created_at, updated_at
        "#,
    )
    .bind(claims.sub)
    .bind(tweet_id)
    .bind(&payload.content)
    .fetch_one(&state.db)
    .await?;

    let (author_username, author_display_name): (String, Option<String>) = sqlx::query_as(
        "SELECT username, display_name FROM users WHERE id = $1"
    )
    .bind(comment.user_id)
    .fetch_one(&state.db)
    .await?;

    Ok((
        StatusCode::CREATED,
        Json(CommentResponse {
            comment,
            author_username,
            author_display_name,
        }),
    ))
}

pub async fn get_comments(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
) -> Result<Json<Vec<CommentResponse>>> {
    let comments = sqlx::query_as::<_, Comment>(
        r#"
        SELECT id, user_id, tweet_id, content, created_at, updated_at
        FROM comments
        WHERE tweet_id = $1
        ORDER BY created_at ASC
        "#,
    )
    .bind(tweet_id)
    .fetch_all(&state.db)
    .await?;

    let mut comment_responses = Vec::new();
    for comment in comments {
        let (author_username, author_display_name): (String, Option<String>) = sqlx::query_as(
            "SELECT username, display_name FROM users WHERE id = $1"
        )
        .bind(comment.user_id)
        .fetch_one(&state.db)
        .await?;

        comment_responses.push(CommentResponse {
            comment,
            author_username,
            author_display_name,
        });
    }

    Ok(Json(comment_responses))
}

pub async fn delete_comment(
    State(state): State<Arc<AppState>>,
    Path(comment_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {

    let comment = sqlx::query_as::<_, Comment>(
        "SELECT id, user_id, tweet_id, content, created_at, updated_at FROM comments WHERE id = $1"
    )
    .bind(comment_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Comment not found".to_string()))?;

    if comment.user_id != claims.sub {
        return Err(AppError::Authorization("Cannot delete another user's comment".to_string()));
    }

    sqlx::query("DELETE FROM comments WHERE id = $1")
        .bind(comment_id)
        .execute(&state.db)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}
