use actix_web::{web, HttpRequest, HttpResponse};
use actix_multipart::Multipart;
use crate::db::AppState;

pub fn get_session_user_id(state: &AppState, req: &HttpRequest) -> Option<i64> {
    todo!()
}

pub async fn register(state: web::Data<AppState>, body: web::Json<serde_json::Value>) -> HttpResponse {
    todo!()
}

pub async fn login(state: web::Data<AppState>, body: web::Json<serde_json::Value>) -> HttpResponse {
    todo!()
}

pub async fn logout(state: web::Data<AppState>, req: HttpRequest) -> HttpResponse {
    todo!()
}

pub async fn me(state: web::Data<AppState>, req: HttpRequest) -> HttpResponse {
    todo!()
}

pub async fn create_post(state: web::Data<AppState>, req: HttpRequest, payload: Multipart) -> HttpResponse {
    todo!()
}

pub async fn get_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn delete_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn list_posts(state: web::Data<AppState>, req: HttpRequest, query: web::Query<serde_json::Value>) -> HttpResponse {
    todo!()
}

pub async fn like_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn unlike_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn follow_user(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn unfollow_user(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn get_followers(state: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn get_following(state: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn get_timeline(state: web::Data<AppState>, req: HttpRequest, query: web::Query<serde_json::Value>) -> HttpResponse {
    todo!()
}

pub async fn get_user_profile(state: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    todo!()
}

pub async fn update_user_profile(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>, body: web::Json<serde_json::Value>) -> HttpResponse {
    todo!()
}

pub async fn get_user_posts(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>, query: web::Query<serde_json::Value>) -> HttpResponse {
    todo!()
}

pub async fn search(state: web::Data<AppState>, req: HttpRequest, query: web::Query<serde_json::Value>) -> HttpResponse {
    todo!()
}

pub async fn hot_topics(state: web::Data<AppState>, req: HttpRequest) -> HttpResponse {
    todo!()
}
