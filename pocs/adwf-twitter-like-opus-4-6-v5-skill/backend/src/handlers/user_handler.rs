use axum::{extract::{Path, State}, http::StatusCode, Extension, Json};

use crate::middleware::auth_middleware::AuthUser;
use crate::models::user::{UpdateUserRequest, User, UserResponse};
use crate::AppState;

pub async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<i32>,
) -> Result<Json<UserResponse>, StatusCode> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = $1")
        .bind(id)
        .fetch_optional(&state.pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(UserResponse::from(user)))
}

pub async fn get_followers(
    State(state): State<AppState>,
    Path(id): Path<i32>,
) -> Result<Json<Vec<UserResponse>>, StatusCode> {
    let users = sqlx::query_as::<_, User>(
        "SELECT u.* FROM users u INNER JOIN follows f ON u.id = f.follower_id WHERE f.following_id = $1"
    )
    .bind(id)
    .fetch_all(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(users.into_iter().map(UserResponse::from).collect()))
}

pub async fn get_following(
    State(state): State<AppState>,
    Path(id): Path<i32>,
) -> Result<Json<Vec<UserResponse>>, StatusCode> {
    let users = sqlx::query_as::<_, User>(
        "SELECT u.* FROM users u INNER JOIN follows f ON u.id = f.following_id WHERE f.follower_id = $1"
    )
    .bind(id)
    .fetch_all(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(users.into_iter().map(UserResponse::from).collect()))
}

pub async fn update_user(
    State(state): State<AppState>,
    Path(id): Path<i32>,
    Extension(auth_user): Extension<AuthUser>,
    Json(payload): Json<UpdateUserRequest>,
) -> Result<Json<UserResponse>, StatusCode> {
    if auth_user.user_id != id {
        return Err(StatusCode::FORBIDDEN);
    }

    let user = sqlx::query_as::<_, User>(
        "UPDATE users SET display_name = COALESCE($1, display_name), bio = COALESCE($2, bio) WHERE id = $3 RETURNING *"
    )
    .bind(&payload.display_name)
    .bind(&payload.bio)
    .bind(id)
    .fetch_one(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(UserResponse::from(user)))
}
