use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use axum::extract::Extension;
use axum::{Json, extract::State};
use sqlx::SqlitePool;

use crate::auth::create_token;
use crate::errors::AppError;
use crate::models::{AuthResponse, Claims, LoginRequest, RegisterRequest, User, UserResponse};

pub async fn register(
    State(pool): State<SqlitePool>,
    Json(body): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, AppError> {
    if body.username.is_empty() || body.email.is_empty() || body.password.is_empty() {
        return Err(AppError::BadRequest("All fields are required".to_string()));
    }

    if !body.email.contains('@') || !body.email.contains('.') {
        return Err(AppError::BadRequest("Invalid email format".to_string()));
    }

    let existing: Option<User> =
        sqlx::query_as("SELECT id, username, email, password_hash, created_at FROM users WHERE email = ?1 OR username = ?2")
            .bind(&body.email)
            .bind(&body.username)
            .fetch_optional(&pool)
            .await?;

    if existing.is_some() {
        return Err(AppError::Conflict(
            "Username or email already exists".to_string(),
        ));
    }

    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(body.password.as_bytes(), &salt)
        .map_err(|e| AppError::Internal(format!("Password hashing failed: {}", e)))?
        .to_string();

    let id = uuid::Uuid::new_v4().to_string();
    let created_at = chrono::Utc::now().to_rfc3339();

    sqlx::query("INSERT INTO users (id, username, email, password_hash, created_at) VALUES (?1, ?2, ?3, ?4, ?5)")
        .bind(&id)
        .bind(&body.username)
        .bind(&body.email)
        .bind(&password_hash)
        .bind(&created_at)
        .execute(&pool)
        .await?;

    let token = create_token(&id)?;

    Ok(Json(AuthResponse {
        token,
        user: UserResponse {
            id,
            username: body.username,
            email: body.email,
            created_at,
        },
    }))
}

pub async fn login(
    State(pool): State<SqlitePool>,
    Json(body): Json<LoginRequest>,
) -> Result<Json<AuthResponse>, AppError> {
    let user: User =
        sqlx::query_as("SELECT id, username, email, password_hash, created_at FROM users WHERE email = ?1")
            .bind(&body.email)
            .fetch_optional(&pool)
            .await?
            .ok_or_else(|| AppError::Unauthorized("Invalid credentials".to_string()))?;

    let parsed_hash = PasswordHash::new(&user.password_hash)
        .map_err(|e| AppError::Internal(format!("Password parse failed: {}", e)))?;

    Argon2::default()
        .verify_password(body.password.as_bytes(), &parsed_hash)
        .map_err(|_| AppError::Unauthorized("Invalid credentials".to_string()))?;

    let token = create_token(&user.id)?;

    Ok(Json(AuthResponse {
        token,
        user: user.into(),
    }))
}

pub async fn me(
    State(pool): State<SqlitePool>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<UserResponse>, AppError> {
    let user: User =
        sqlx::query_as("SELECT id, username, email, password_hash, created_at FROM users WHERE id = ?1")
            .bind(&claims.sub)
            .fetch_optional(&pool)
            .await?
            .ok_or_else(|| AppError::NotFound("User not found".to_string()))?;

    Ok(Json(user.into()))
}
