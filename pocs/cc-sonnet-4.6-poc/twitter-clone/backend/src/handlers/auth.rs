use actix_web::{web, HttpResponse, Responder};
use bcrypt::{hash, verify, DEFAULT_COST};
use sqlx::Row;
use crate::models::{LoginRequest, RegisterRequest, AuthResponse};
use crate::{AppState, auth};

pub async fn register(
    state: web::Data<AppState>,
    body: web::Json<RegisterRequest>,
) -> impl Responder {
    let password_hash = match hash(&body.password, DEFAULT_COST) {
        Ok(h) => h,
        Err(_) => return HttpResponse::InternalServerError().json("Failed to hash password"),
    };

    let result = sqlx::query(
        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
    )
    .bind(&body.username)
    .bind(&body.email)
    .bind(&password_hash)
    .execute(&state.db)
    .await;

    match result {
        Ok(r) => {
            let user_id = r.last_insert_rowid();
            let token = match auth::create_token(user_id, &body.username, &state.jwt_secret) {
                Some(t) => t,
                None => return HttpResponse::InternalServerError().json("Failed to create token"),
            };
            HttpResponse::Ok().json(AuthResponse {
                token,
                user_id,
                username: body.username.clone(),
            })
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("UNIQUE") {
                HttpResponse::Conflict().json("Username or email already exists")
            } else {
                HttpResponse::InternalServerError().json("Registration failed")
            }
        }
    }
}

pub async fn login(
    state: web::Data<AppState>,
    body: web::Json<LoginRequest>,
) -> impl Responder {
    let user = sqlx::query(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
    )
    .bind(&body.username)
    .fetch_optional(&state.db)
    .await;

    match user {
        Ok(Some(row)) => {
            let id: i64 = row.get("id");
            let username: String = row.get("username");
            let password_hash: String = row.get("password_hash");

            let valid = verify(&body.password, &password_hash).unwrap_or(false);
            if !valid {
                return HttpResponse::Unauthorized().json("Invalid credentials");
            }
            let token = match auth::create_token(id, &username, &state.jwt_secret) {
                Some(t) => t,
                None => return HttpResponse::InternalServerError().json("Failed to create token"),
            };
            HttpResponse::Ok().json(AuthResponse {
                token,
                user_id: id,
                username,
            })
        }
        Ok(None) => HttpResponse::Unauthorized().json("Invalid credentials"),
        Err(_) => HttpResponse::InternalServerError().json("Login failed"),
    }
}
