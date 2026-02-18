use actix_web::{web, HttpRequest};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use crate::models::Claims;
use crate::AppState;

pub fn create_token(user_id: i64, username: &str, secret: &str) -> Option<String> {
    let expiration = chrono::Utc::now()
        .checked_add_signed(chrono::Duration::hours(24))
        .map(|t| t.timestamp() as usize)?;

    let claims = Claims {
        sub: username.to_string(),
        user_id,
        exp: expiration,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .ok()
}

pub fn extract_user_id(req: &HttpRequest, state: &web::Data<AppState>) -> Option<i64> {
    let auth_header = req.headers().get("Authorization")?;
    let auth_str = auth_header.to_str().ok()?;

    if !auth_str.starts_with("Bearer ") {
        return None;
    }

    let token = &auth_str[7..];

    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(state.jwt_secret.as_bytes()),
        &Validation::default(),
    )
    .ok()?;

    Some(token_data.claims.user_id)
}
