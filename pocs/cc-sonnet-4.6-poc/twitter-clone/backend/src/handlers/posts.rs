use actix_web::{web, HttpRequest, HttpResponse, Responder};
use sqlx::Row;
use crate::models::{CreatePostRequest, PostWithUser};
use crate::{AppState, auth};

pub async fn create_post(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<CreatePostRequest>,
) -> impl Responder {
    let user_id = match auth::extract_user_id(&req, &state) {
        Some(id) => id,
        None => return HttpResponse::Unauthorized().json("Not authenticated"),
    };

    if body.content.trim().is_empty() || body.content.len() > 280 {
        return HttpResponse::BadRequest().json("Content must be 1-280 characters");
    }

    let result = sqlx::query("INSERT INTO posts (user_id, content) VALUES (?, ?)")
        .bind(user_id)
        .bind(&body.content)
        .execute(&state.db)
        .await;

    match result {
        Ok(r) => {
            let post_id = r.last_insert_rowid();
            HttpResponse::Ok().json(serde_json::json!({ "id": post_id, "content": body.content }))
        }
        Err(_) => HttpResponse::InternalServerError().json("Failed to create post"),
    }
}

pub async fn timeline(
    req: HttpRequest,
    state: web::Data<AppState>,
) -> impl Responder {
    let user_id = match auth::extract_user_id(&req, &state) {
        Some(id) => id,
        None => return HttpResponse::Unauthorized().json("Not authenticated"),
    };

    let rows = sqlx::query(
        r#"SELECT p.id, p.content, p.created_at, p.user_id, u.username,
            (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as likes_count,
            (SELECT COUNT(*) FROM likes WHERE post_id = p.id AND user_id = ?) as liked_by_me
           FROM posts p
           JOIN users u ON p.user_id = u.id
           WHERE p.user_id = ? OR p.user_id IN (
               SELECT following_id FROM follows WHERE follower_id = ?
           )
           ORDER BY p.created_at DESC
           LIMIT 100"#,
    )
    .bind(user_id)
    .bind(user_id)
    .bind(user_id)
    .fetch_all(&state.db)
    .await;

    match rows {
        Ok(r) => {
            let posts: Vec<PostWithUser> = r
                .into_iter()
                .map(|row| PostWithUser {
                    id: row.get("id"),
                    content: row.get("content"),
                    created_at: row.get("created_at"),
                    user_id: row.get("user_id"),
                    username: row.get("username"),
                    likes_count: row.get::<i64, _>("likes_count"),
                    liked_by_me: row.get::<i64, _>("liked_by_me") > 0,
                })
                .collect();
            HttpResponse::Ok().json(posts)
        }
        Err(_) => HttpResponse::InternalServerError().json("Failed to fetch timeline"),
    }
}

pub async fn user_posts(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<i64>,
) -> impl Responder {
    let current_user_id = auth::extract_user_id(&req, &state).unwrap_or(-1);
    let target_user_id = path.into_inner();

    let rows = sqlx::query(
        r#"SELECT p.id, p.content, p.created_at, p.user_id, u.username,
            (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as likes_count,
            (SELECT COUNT(*) FROM likes WHERE post_id = p.id AND user_id = ?) as liked_by_me
           FROM posts p
           JOIN users u ON p.user_id = u.id
           WHERE p.user_id = ?
           ORDER BY p.created_at DESC
           LIMIT 50"#,
    )
    .bind(current_user_id)
    .bind(target_user_id)
    .fetch_all(&state.db)
    .await;

    match rows {
        Ok(r) => {
            let posts: Vec<PostWithUser> = r
                .into_iter()
                .map(|row| PostWithUser {
                    id: row.get("id"),
                    content: row.get("content"),
                    created_at: row.get("created_at"),
                    user_id: row.get("user_id"),
                    username: row.get("username"),
                    likes_count: row.get::<i64, _>("likes_count"),
                    liked_by_me: row.get::<i64, _>("liked_by_me") > 0,
                })
                .collect();
            HttpResponse::Ok().json(posts)
        }
        Err(_) => HttpResponse::InternalServerError().json("Failed to fetch posts"),
    }
}
