use crate::models::types::{AppState, TrackRequest};
use crate::persistence::db;
use actix_web::{HttpResponse, web};
use chrono::{Datelike, Local};

pub async fn get_weekly(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let username = path.into_inner();
    let conn = data.db.lock().await;
    match db::get_weekly_scores(&conn, &username) {
        Ok(scores) => HttpResponse::Ok().json(scores),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn get_leaderboard(data: web::Data<AppState>) -> HttpResponse {
    let today = Local::now().date_naive();
    let days_since_monday = today.weekday().num_days_from_monday();
    let monday = today - chrono::Duration::days(days_since_monday as i64);
    let week_start = monday.format("%Y-%m-%d").to_string();

    let conn = data.db.lock().await;
    match db::get_leaderboard(&conn, &week_start) {
        Ok(entries) => HttpResponse::Ok().json(entries),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn get_leaderboard_week(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let week_start = path.into_inner();
    let conn = data.db.lock().await;
    match db::get_leaderboard(&conn, &week_start) {
        Ok(entries) => HttpResponse::Ok().json(entries),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn get_tracked_users(data: web::Data<AppState>) -> HttpResponse {
    let conn = data.db.lock().await;
    match db::get_tracked_users(&conn) {
        Ok(users) => HttpResponse::Ok().json(users),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn track_user(
    body: web::Json<TrackRequest>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let conn = data.db.lock().await;
    match db::insert_tracked_user(&conn, &body.github_user) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({"status": "ok"})),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn untrack_user(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let username = path.into_inner();
    let conn = data.db.lock().await;
    match db::delete_tracked_user(&conn, &username) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({"status": "ok"})),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}
