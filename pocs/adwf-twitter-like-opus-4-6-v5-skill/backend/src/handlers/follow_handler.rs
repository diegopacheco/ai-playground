use axum::{extract::{Path, State}, http::StatusCode, Extension, Json};
use serde::Serialize;

use crate::middleware::auth_middleware::AuthUser;
use crate::AppState;

#[derive(Serialize)]
pub struct FollowResponse {
    pub following: bool,
}

pub async fn follow_user(
    State(state): State<AppState>,
    Path(id): Path<i32>,
    Extension(auth_user): Extension<AuthUser>,
) -> Result<Json<FollowResponse>, StatusCode> {
    if auth_user.user_id == id {
        return Err(StatusCode::BAD_REQUEST);
    }

    sqlx::query("INSERT INTO follows (follower_id, following_id) VALUES ($1, $2) ON CONFLICT DO NOTHING")
        .bind(auth_user.user_id)
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(FollowResponse { following: true }))
}

pub async fn unfollow_user(
    State(state): State<AppState>,
    Path(id): Path<i32>,
    Extension(auth_user): Extension<AuthUser>,
) -> Result<Json<FollowResponse>, StatusCode> {
    sqlx::query("DELETE FROM follows WHERE follower_id = $1 AND following_id = $2")
        .bind(auth_user.user_id)
        .bind(id)
        .execute(&state.pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(FollowResponse { following: false }))
}
