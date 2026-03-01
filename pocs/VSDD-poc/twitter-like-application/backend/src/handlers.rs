use actix_web::{web, HttpRequest, HttpResponse, cookie::Cookie};
use actix_multipart::Multipart;
use futures_util::StreamExt;
use crate::db::AppState;
use crate::validation;

pub fn get_session_user_id(state: &AppState, req: &HttpRequest) -> Option<i64> {
    let cookie = req.cookie("session_id")?;
    let sessions = state.sessions.lock().unwrap();
    sessions.get(cookie.value()).copied()
}

fn require_auth(state: &AppState, req: &HttpRequest) -> Result<i64, HttpResponse> {
    get_session_user_id(state, req).ok_or_else(|| {
        HttpResponse::Unauthorized().json(serde_json::json!({"error": "Not authenticated"}))
    })
}

fn error_response(code: u16, msg: &str) -> HttpResponse {
    let status = actix_web::http::StatusCode::from_u16(code).unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
    HttpResponse::build(status).json(serde_json::json!({"error": msg}))
}

fn parse_page_limit(query: &serde_json::Value) -> (i64, i64) {
    let page = query.get("page").and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(1i64).max(1);
    let limit = query.get("limit").and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(20i64).clamp(1, 100);
    (page, limit)
}

pub async fn register(state: web::Data<AppState>, body: web::Json<serde_json::Value>) -> HttpResponse {
    let username = body.get("username").and_then(|v| v.as_str()).unwrap_or("");
    let password = body.get("password").and_then(|v| v.as_str()).unwrap_or("");
    let display_name = body.get("display_name").and_then(|v| v.as_str()).unwrap_or("");
    if let Err(e) = validation::validate_username(username) {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
    }
    if let Err(e) = validation::validate_password(password) {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
    }
    if let Err(e) = validation::validate_display_name(display_name) {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
    }
    match crate::db::register_user(&state, username, password, display_name) {
        Ok(result) => {
            let session_id = result["session_id"].as_str().unwrap().to_string();
            let cookie = Cookie::build("session_id", session_id)
                .path("/")
                .http_only(true)
                .finish();
            HttpResponse::Created()
                .cookie(cookie)
                .json(result["user"].clone())
        }
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn login(state: web::Data<AppState>, body: web::Json<serde_json::Value>) -> HttpResponse {
    let username = body.get("username").and_then(|v| v.as_str()).unwrap_or("");
    let password = body.get("password").and_then(|v| v.as_str()).unwrap_or("");
    match crate::db::login_user(&state, username, password) {
        Ok((user, session_id)) => {
            let cookie = Cookie::build("session_id", session_id)
                .path("/")
                .http_only(true)
                .finish();
            HttpResponse::Ok().cookie(cookie).json(user)
        }
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn logout(state: web::Data<AppState>, req: HttpRequest) -> HttpResponse {
    if let Some(cookie) = req.cookie("session_id") {
        state.sessions.lock().unwrap().remove(cookie.value());
    }
    let mut removal = Cookie::build("session_id", "")
        .path("/")
        .http_only(true)
        .finish();
    removal.make_removal();
    HttpResponse::Ok().cookie(removal).json(serde_json::json!({"ok": true}))
}

pub async fn me(state: web::Data<AppState>, req: HttpRequest) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let conn = state.db.lock().unwrap();
    match crate::db::get_user_by_id(&conn, user_id) {
        Ok(user) => HttpResponse::Ok().json(user),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn create_post(state: web::Data<AppState>, req: HttpRequest, mut payload: Multipart) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let mut content = String::new();
    let mut image_url = String::new();
    while let Some(Ok(mut field)) = payload.next().await {
        let field_name = field.name().map(|s| s.to_string()).unwrap_or_default();
        match field_name.as_str() {
            "content" => {
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = field.next().await {
                    bytes.extend_from_slice(&chunk);
                }
                content = String::from_utf8_lossy(&bytes).to_string();
            }
            "image" => {
                let content_type = field.content_type().map(|m| m.to_string()).unwrap_or_default();
                if let Err(e) = validation::validate_image_type(&content_type) {
                    return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
                }
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = field.next().await {
                    bytes.extend_from_slice(&chunk);
                }
                if let Err(e) = validation::validate_image_size(bytes.len()) {
                    return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
                }
                let ext = if content_type == "image/png" { "png" } else { "jpg" };
                let filename = format!("{}.{}", uuid::Uuid::new_v4(), ext);
                let path = format!("uploads/{}", filename);
                if let Err(_) = std::fs::write(&path, &bytes) {
                    return HttpResponse::InternalServerError().json(serde_json::json!({"error": "Failed to save image"}));
                }
                image_url = filename;
            }
            _ => {}
        }
    }
    if let Err(e) = validation::validate_post_content(&content) {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
    }
    let conn = state.db.lock().unwrap();
    match crate::db::create_post(&conn, user_id, &content, &image_url) {
        Ok(post) => HttpResponse::Created().json(post),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn get_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    let uid = get_session_user_id(&state, &req);
    let conn = state.db.lock().unwrap();
    match crate::db::get_post(&conn, path.into_inner(), uid) {
        Ok(post) => HttpResponse::Ok().json(post),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn delete_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let conn = state.db.lock().unwrap();
    match crate::db::delete_post(&conn, path.into_inner(), user_id) {
        Ok(image_url) => {
            if !image_url.is_empty() {
                let _ = std::fs::remove_file(format!("uploads/{}", image_url));
            }
            HttpResponse::NoContent().finish()
        }
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn list_posts(state: web::Data<AppState>, req: HttpRequest, query: web::Query<serde_json::Value>) -> HttpResponse {
    let uid = get_session_user_id(&state, &req);
    let (page, limit) = parse_page_limit(&query);
    let conn = state.db.lock().unwrap();
    match crate::db::list_posts(&conn, page, limit, uid) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn like_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let conn = state.db.lock().unwrap();
    match crate::db::like_post(&conn, user_id, path.into_inner()) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn unlike_post(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let conn = state.db.lock().unwrap();
    match crate::db::unlike_post(&conn, user_id, path.into_inner()) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn follow_user(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let conn = state.db.lock().unwrap();
    match crate::db::follow_user(&conn, user_id, path.into_inner()) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({"ok": true})),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn unfollow_user(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let conn = state.db.lock().unwrap();
    match crate::db::unfollow_user(&conn, user_id, path.into_inner()) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({"ok": true})),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn get_followers(state: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let conn = state.db.lock().unwrap();
    match crate::db::get_followers(&conn, path.into_inner()) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn get_following(state: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let conn = state.db.lock().unwrap();
    match crate::db::get_following(&conn, path.into_inner()) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn get_timeline(state: web::Data<AppState>, req: HttpRequest, query: web::Query<serde_json::Value>) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let (page, limit) = parse_page_limit(&query);
    let conn = state.db.lock().unwrap();
    match crate::db::get_timeline(&conn, user_id, page, limit) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn get_user_profile(state: web::Data<AppState>, path: web::Path<i64>) -> HttpResponse {
    let conn = state.db.lock().unwrap();
    match crate::db::get_user_profile(&conn, path.into_inner()) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn update_user_profile(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>, body: web::Json<serde_json::Value>) -> HttpResponse {
    let user_id = match require_auth(&state, &req) {
        Ok(id) => id,
        Err(resp) => return resp,
    };
    let target_id = path.into_inner();
    if user_id != target_id {
        return HttpResponse::Forbidden().json(serde_json::json!({"error": "Cannot update another user's profile"}));
    }
    let display_name = body.get("display_name").and_then(|v| v.as_str()).unwrap_or("");
    let bio = body.get("bio").and_then(|v| v.as_str()).unwrap_or("");
    if let Err(e) = validation::validate_display_name(display_name) {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
    }
    if let Err(e) = validation::validate_bio(bio) {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": e}));
    }
    let conn = state.db.lock().unwrap();
    match crate::db::update_user_profile(&conn, user_id, display_name, bio) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn get_user_posts(state: web::Data<AppState>, req: HttpRequest, path: web::Path<i64>, query: web::Query<serde_json::Value>) -> HttpResponse {
    let uid = get_session_user_id(&state, &req);
    let (page, limit) = parse_page_limit(&query);
    let conn = state.db.lock().unwrap();
    match crate::db::get_user_posts(&conn, path.into_inner(), page, limit, uid) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn search(state: web::Data<AppState>, req: HttpRequest, query: web::Query<serde_json::Value>) -> HttpResponse {
    let q = query.get("q").and_then(|v| v.as_str()).unwrap_or("");
    let search_type = query.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if q.is_empty() {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": "Search query cannot be empty"}));
    }
    if search_type != "posts" && search_type != "users" {
        return HttpResponse::BadRequest().json(serde_json::json!({"error": "Type must be 'posts' or 'users'"}));
    }
    let (page, limit) = parse_page_limit(&query);
    let conn = state.db.lock().unwrap();
    let result = if search_type == "posts" {
        let uid = get_session_user_id(&state, &req);
        crate::db::search_posts(&conn, q, page, limit, uid)
    } else {
        crate::db::search_users(&conn, q, page, limit)
    };
    match result {
        Ok(r) => HttpResponse::Ok().json(r),
        Err((code, msg)) => error_response(code, &msg),
    }
}

pub async fn hot_topics(state: web::Data<AppState>, req: HttpRequest) -> HttpResponse {
    let uid = get_session_user_id(&state, &req);
    let conn = state.db.lock().unwrap();
    match crate::db::get_hot_posts(&conn, uid) {
        Ok(result) => HttpResponse::Ok().json(result),
        Err((code, msg)) => error_response(code, &msg),
    }
}
