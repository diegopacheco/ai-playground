use axum::{extract::State, http::StatusCode, Json};

use crate::auth::create_token;
use crate::models::user::{AuthResponse, LoginRequest, RegisterRequest, User, UserResponse};
use crate::AppState;

pub async fn register(
    State(state): State<AppState>,
    Json(payload): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, StatusCode> {
    let password_hash = bcrypt::hash(&payload.password, 10)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let user = sqlx::query_as::<_, User>(
        "INSERT INTO users (username, email, password_hash, display_name) VALUES ($1, $2, $3, $4) RETURNING *"
    )
    .bind(&payload.username)
    .bind(&payload.email)
    .bind(&password_hash)
    .bind(&payload.username)
    .fetch_one(&state.pool)
    .await
    .map_err(|_| StatusCode::CONFLICT)?;

    let token = create_token(user.id, &state.jwt_secret)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(AuthResponse {
        token,
        user: UserResponse::from(user),
    }))
}

pub async fn login(
    State(state): State<AppState>,
    Json(payload): Json<LoginRequest>,
) -> Result<Json<AuthResponse>, StatusCode> {
    let user = sqlx::query_as::<_, User>(
        "SELECT * FROM users WHERE email = $1"
    )
    .bind(&payload.email)
    .fetch_optional(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .ok_or(StatusCode::UNAUTHORIZED)?;

    let valid = bcrypt::verify(&payload.password, &user.password_hash)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if !valid {
        return Err(StatusCode::UNAUTHORIZED);
    }

    let token = create_token(user.id, &state.jwt_secret)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(AuthResponse {
        token,
        user: UserResponse::from(user),
    }))
}
