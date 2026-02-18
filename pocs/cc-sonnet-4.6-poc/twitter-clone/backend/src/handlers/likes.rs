use actix_web::{web, HttpRequest, HttpResponse, Responder};
use crate::{AppState, auth};

pub async fn like_post(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<i64>,
) -> impl Responder {
    let user_id = match auth::extract_user_id(&req, &state) {
        Some(id) => id,
        None => return HttpResponse::Unauthorized().json("Not authenticated"),
    };
    let post_id = path.into_inner();

    let result = sqlx::query("INSERT OR IGNORE INTO likes (user_id, post_id) VALUES (?, ?)")
        .bind(user_id)
        .bind(post_id)
        .execute(&state.db)
        .await;

    match result {
        Ok(_) => HttpResponse::Ok().json("Liked"),
        Err(_) => HttpResponse::InternalServerError().json("Failed to like"),
    }
}

pub async fn unlike_post(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<i64>,
) -> impl Responder {
    let user_id = match auth::extract_user_id(&req, &state) {
        Some(id) => id,
        None => return HttpResponse::Unauthorized().json("Not authenticated"),
    };
    let post_id = path.into_inner();

    let result = sqlx::query("DELETE FROM likes WHERE user_id = ? AND post_id = ?")
        .bind(user_id)
        .bind(post_id)
        .execute(&state.db)
        .await;

    match result {
        Ok(_) => HttpResponse::Ok().json("Unliked"),
        Err(_) => HttpResponse::InternalServerError().json("Failed to unlike"),
    }
}
