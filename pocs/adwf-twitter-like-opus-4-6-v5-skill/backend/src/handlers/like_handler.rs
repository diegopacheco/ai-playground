use axum::{extract::{Path, State}, http::StatusCode, Extension, Json};
use serde::Serialize;

use crate::middleware::auth_middleware::AuthUser;
use crate::AppState;

#[derive(Serialize)]
pub struct LikeResponse {
    pub liked: bool,
}

pub async fn like_tweet(
    State(state): State<AppState>,
    Path(id): Path<i32>,
    Extension(auth_user): Extension<AuthUser>,
) -> Result<Json<LikeResponse>, StatusCode> {
    sqlx::query("INSERT INTO likes (user_id, tweet_id) VALUES ($1, $2) ON CONFLICT DO NOTHING")
        .bind(auth_user.user_id)
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(LikeResponse { liked: true }))
}

pub async fn unlike_tweet(
    State(state): State<AppState>,
    Path(id): Path<i32>,
    Extension(auth_user): Extension<AuthUser>,
) -> Result<Json<LikeResponse>, StatusCode> {
    sqlx::query("DELETE FROM likes WHERE user_id = $1 AND tweet_id = $2")
        .bind(auth_user.user_id)
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(LikeResponse { liked: false }))
}
