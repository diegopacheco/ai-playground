use axum::extract::{Path, State};
use axum::Json;
use sqlx::SqlitePool;

use crate::auth::{create_token, AuthUser};
use crate::errors::AppError;
use crate::models::user::{
    CreateUserRequest, LoginRequest, LoginResponse, User, UserResponse,
};

pub async fn register(
    State(pool): State<SqlitePool>,
    Json(req): Json<CreateUserRequest>,
) -> Result<Json<UserResponse>, AppError> {
    if req.username.is_empty() || req.email.is_empty() || req.password.is_empty() {
        return Err(AppError::BadRequest("All fields are required".to_string()));
    }

    let password_hash = bcrypt::hash(&req.password, 12)?;

    let result = sqlx::query(
        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)"
    )
    .bind(&req.username)
    .bind(&req.email)
    .bind(&password_hash)
    .execute(&pool)
    .await;

    match result {
        Ok(res) => {
            let user_id = res.last_insert_rowid();
            let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?")
                .bind(user_id)
                .fetch_one(&pool)
                .await?;
            Ok(Json(UserResponse::from(user)))
        }
        Err(sqlx::Error::Database(err)) if err.message().contains("UNIQUE") => {
            Err(AppError::Conflict("Username or email already exists".to_string()))
        }
        Err(err) => Err(AppError::from(err)),
    }
}

pub async fn login(
    State(pool): State<SqlitePool>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AppError> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE username = ?")
        .bind(&req.username)
        .fetch_optional(&pool)
        .await?
        .ok_or_else(|| AppError::Unauthorized("Invalid credentials".to_string()))?;

    let valid = bcrypt::verify(&req.password, &user.password_hash)?;
    if !valid {
        return Err(AppError::Unauthorized("Invalid credentials".to_string()));
    }

    let token = create_token(user.id, &user.username)?;

    Ok(Json(LoginResponse {
        token,
        user: UserResponse::from(user),
    }))
}

pub async fn get_user(
    State(pool): State<SqlitePool>,
    Path(id): Path<i64>,
) -> Result<Json<UserResponse>, AppError> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?")
        .bind(id)
        .fetch_optional(&pool)
        .await?
        .ok_or_else(|| AppError::NotFound("User not found".to_string()))?;

    Ok(Json(UserResponse::from(user)))
}

pub async fn get_followers(
    State(pool): State<SqlitePool>,
    Path(id): Path<i64>,
) -> Result<Json<Vec<UserResponse>>, AppError> {
    let users = sqlx::query_as::<_, User>(
        "SELECT u.* FROM users u
         INNER JOIN follows f ON f.follower_id = u.id
         WHERE f.following_id = ?
         ORDER BY f.created_at DESC"
    )
    .bind(id)
    .fetch_all(&pool)
    .await?;

    let response: Vec<UserResponse> = users.into_iter().map(UserResponse::from).collect();
    Ok(Json(response))
}

pub async fn get_following(
    State(pool): State<SqlitePool>,
    Path(id): Path<i64>,
) -> Result<Json<Vec<UserResponse>>, AppError> {
    let users = sqlx::query_as::<_, User>(
        "SELECT u.* FROM users u
         INNER JOIN follows f ON f.following_id = u.id
         WHERE f.follower_id = ?
         ORDER BY f.created_at DESC"
    )
    .bind(id)
    .fetch_all(&pool)
    .await?;

    let response: Vec<UserResponse> = users.into_iter().map(UserResponse::from).collect();
    Ok(Json(response))
}

pub async fn get_me(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
) -> Result<Json<UserResponse>, AppError> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?")
        .bind(auth.user_id)
        .fetch_optional(&pool)
        .await?
        .ok_or_else(|| AppError::NotFound("User not found".to_string()))?;

    Ok(Json(UserResponse::from(user)))
}
