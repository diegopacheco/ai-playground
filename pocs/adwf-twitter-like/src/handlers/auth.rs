use axum::{extract::State, http::StatusCode, Json};
use bcrypt::{hash, verify, DEFAULT_COST};
use std::sync::Arc;
use validator::Validate;

use crate::{
    error::{AppError, Result},
    middleware::create_jwt,
    models::{AuthResponse, LoginRequest, RegisterRequest, User},
    state::AppState,
};

pub async fn register(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RegisterRequest>,
) -> Result<(StatusCode, Json<AuthResponse>)> {
    payload.validate().map_err(|e| AppError::BadRequest(e.to_string()))?;

    let password_hash = hash(&payload.password, DEFAULT_COST)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to hash password: {}", e)))?;

    let user = sqlx::query_as::<_, User>(
        r#"
        INSERT INTO users (username, email, password_hash)
        VALUES ($1, $2, $3)
        RETURNING id, username, email, password_hash, display_name, bio, created_at, updated_at
        "#,
    )
    .bind(&payload.username)
    .bind(&payload.email)
    .bind(&password_hash)
    .fetch_one(&state.db)
    .await
    .map_err(|e| match e {
        sqlx::Error::Database(db_err) if db_err.is_unique_violation() => {
            AppError::BadRequest("Username or email already exists".to_string())
        }
        _ => AppError::Database(e),
    })?;

    let token = create_jwt(user.id, &state.config.jwt_secret)?;

    Ok((
        StatusCode::CREATED,
        Json(AuthResponse { token, user }),
    ))
}

pub async fn login(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LoginRequest>,
) -> Result<Json<AuthResponse>> {
    let user = sqlx::query_as::<_, User>(
        r#"
        SELECT id, username, email, password_hash, display_name, bio, created_at, updated_at
        FROM users
        WHERE username = $1 OR email = $1
        "#,
    )
    .bind(&payload.username)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::Authentication("Invalid username or password".to_string()))?;

    let valid = verify(&payload.password, &user.password_hash)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to verify password: {}", e)))?;

    if !valid {
        return Err(AppError::Authentication("Invalid username or password".to_string()));
    }

    let token = create_jwt(user.id, &state.config.jwt_secret)?;

    Ok(Json(AuthResponse { token, user }))
}

pub async fn logout() -> Result<StatusCode> {
    Ok(StatusCode::NO_CONTENT)
}
