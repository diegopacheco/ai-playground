use axum::{
    extract::{Path, Request, State},
    http::StatusCode,
    Extension,
    Json,
};
use std::sync::Arc;
use validator::Validate;

use crate::{
    error::{AppError, Result},
    middleware::Claims,
    models::{UpdateUserRequest, User, UserProfile},
    state::AppState,
};

pub async fn get_user_profile(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i32>,
    request: Request,
) -> Result<Json<UserProfile>> {
    let _current_user_id = request
        .extensions()
        .get::<Claims>()
        .map(|claims| claims.sub);

    let user = sqlx::query_as::<_, User>(
        r#"
        SELECT id, username, email, password_hash, display_name, bio, created_at, updated_at
        FROM users
        WHERE id = $1
        "#,
    )
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("User not found".to_string()))?;

    let followers_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM follows WHERE following_id = $1"
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;

    let following_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM follows WHERE follower_id = $1"
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;

    let tweets_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tweets WHERE user_id = $1"
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;

    Ok(Json(UserProfile {
        user,
        followers_count,
        following_count,
        tweets_count,
    }))
}

pub async fn update_user_profile(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i32>,
    Extension(claims): Extension<Claims>,
    Json(payload): Json<UpdateUserRequest>,
) -> Result<Json<User>> {
    payload.validate().map_err(|e| AppError::BadRequest(e.to_string()))?;

    if claims.sub != user_id {
        return Err(AppError::Authorization("Cannot update another user's profile".to_string()));
    }

    let user = sqlx::query_as::<_, User>(
        r#"
        UPDATE users
        SET display_name = COALESCE($1, display_name),
            bio = COALESCE($2, bio),
            updated_at = NOW()
        WHERE id = $3
        RETURNING id, username, email, password_hash, display_name, bio, created_at, updated_at
        "#,
    )
    .bind(&payload.display_name)
    .bind(&payload.bio)
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;

    Ok(Json(user))
}

pub async fn get_followers(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i32>,
) -> Result<Json<Vec<User>>> {
    let followers = sqlx::query_as::<_, User>(
        r#"
        SELECT u.id, u.username, u.email, u.password_hash, u.display_name, u.bio, u.created_at, u.updated_at
        FROM users u
        INNER JOIN follows f ON f.follower_id = u.id
        WHERE f.following_id = $1
        ORDER BY f.created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(&state.db)
    .await?;

    Ok(Json(followers))
}

pub async fn get_following(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i32>,
) -> Result<Json<Vec<User>>> {
    let following = sqlx::query_as::<_, User>(
        r#"
        SELECT u.id, u.username, u.email, u.password_hash, u.display_name, u.bio, u.created_at, u.updated_at
        FROM users u
        INNER JOIN follows f ON f.following_id = u.id
        WHERE f.follower_id = $1
        ORDER BY f.created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(&state.db)
    .await?;

    Ok(Json(following))
}

pub async fn follow_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {

    if claims.sub == user_id {
        return Err(AppError::BadRequest("Cannot follow yourself".to_string()));
    }

    sqlx::query(
        r#"
        INSERT INTO follows (follower_id, following_id)
        VALUES ($1, $2)
        ON CONFLICT DO NOTHING
        "#,
    )
    .bind(claims.sub)
    .bind(user_id)
    .execute(&state.db)
    .await?;

    Ok(StatusCode::CREATED)
}

pub async fn unfollow_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {

    sqlx::query(
        r#"
        DELETE FROM follows
        WHERE follower_id = $1 AND following_id = $2
        "#,
    )
    .bind(claims.sub)
    .bind(user_id)
    .execute(&state.db)
    .await?;

    Ok(StatusCode::NO_CONTENT)
}
