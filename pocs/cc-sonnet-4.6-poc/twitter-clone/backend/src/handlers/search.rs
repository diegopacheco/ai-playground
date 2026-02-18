use actix_web::{web, HttpRequest, HttpResponse, Responder};
use sqlx::Row;
use crate::models::{PostWithUser, SearchQuery};
use crate::{AppState, auth};

pub async fn search(
    req: HttpRequest,
    state: web::Data<AppState>,
    query: web::Query<SearchQuery>,
) -> impl Responder {
    let current_user_id = auth::extract_user_id(&req, &state).unwrap_or(-1);
    let search_term = format!("%{}%", query.q.trim());

    let posts_rows = sqlx::query(
        r#"SELECT p.id, p.content, p.created_at, p.user_id, u.username,
            (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as likes_count,
            (SELECT COUNT(*) FROM likes WHERE post_id = p.id AND user_id = ?) as liked_by_me
           FROM posts p
           JOIN users u ON p.user_id = u.id
           WHERE p.content LIKE ? OR u.username LIKE ?
           ORDER BY p.created_at DESC
           LIMIT 50"#,
    )
    .bind(current_user_id)
    .bind(&search_term)
    .bind(&search_term)
    .fetch_all(&state.db)
    .await;

    let users_rows = sqlx::query(
        "SELECT id, username, bio FROM users WHERE username LIKE ? LIMIT 10",
    )
    .bind(&search_term)
    .fetch_all(&state.db)
    .await;

    match (posts_rows, users_rows) {
        (Ok(p), Ok(u)) => {
            let post_results: Vec<PostWithUser> = p
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

            let user_results: Vec<serde_json::Value> = u
                .into_iter()
                .map(|row| {
                    let id: i64 = row.get("id");
                    let username: String = row.get("username");
                    let bio: Option<String> = row.get("bio");
                    serde_json::json!({ "id": id, "username": username, "bio": bio })
                })
                .collect();

            HttpResponse::Ok().json(serde_json::json!({
                "posts": post_results,
                "users": user_results
            }))
        }
        _ => HttpResponse::InternalServerError().json("Search failed"),
    }
}
