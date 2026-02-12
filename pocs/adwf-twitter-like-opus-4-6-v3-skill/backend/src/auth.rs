use axum::extract::Request;
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::Response;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};

use crate::errors::AppError;
use crate::models::Claims;

const JWT_SECRET: &str = "twitter-clone-secret-key-change-in-production";

pub fn create_token(user_id: &str) -> Result<String, AppError> {
    let expiration = chrono::Utc::now()
        .checked_add_signed(chrono::Duration::hours(24))
        .expect("valid timestamp")
        .timestamp() as usize;

    let claims = Claims {
        sub: user_id.to_string(),
        exp: expiration,
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(JWT_SECRET.as_bytes()),
    )?;

    Ok(token)
}

pub fn verify_token(token: &str) -> Result<Claims, AppError> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(JWT_SECRET.as_bytes()),
        &Validation::default(),
    )?;

    Ok(token_data.claims)
}

pub async fn auth_middleware(mut req: Request, next: Next) -> Result<Response, AppError> {
    let auth_header = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| AppError::Unauthorized("Missing authorization header".to_string()))?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| AppError::Unauthorized("Invalid authorization format".to_string()))?;

    let claims = verify_token(token)?;
    req.extensions_mut().insert(claims);

    Ok(next.run(req).await)
}
