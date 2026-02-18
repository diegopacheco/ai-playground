use actix_web::{web, HttpRequest, HttpResponse, Responder};
use sqlx::Row;
use crate::models::UserProfile;
use crate::{AppState, auth};

pub async fn follow_user(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<i64>,
) -> impl Responder {
    let follower_id = match auth::extract_user_id(&req, &state) {
        Some(id) => id,
        None => return HttpResponse::Unauthorized().json("Not authenticated"),
    };
    let following_id = path.into_inner();

    if follower_id == following_id {
        return HttpResponse::BadRequest().json("Cannot follow yourself");
    }

    let result = sqlx::query(
        "INSERT OR IGNORE INTO follows (follower_id, following_id) VALUES (?, ?)",
    )
    .bind(follower_id)
    .bind(following_id)
    .execute(&state.db)
    .await;

    match result {
        Ok(_) => HttpResponse::Ok().json("Followed"),
        Err(_) => HttpResponse::InternalServerError().json("Failed to follow"),
    }
}

pub async fn unfollow_user(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<i64>,
) -> impl Responder {
    let follower_id = match auth::extract_user_id(&req, &state) {
        Some(id) => id,
        None => return HttpResponse::Unauthorized().json("Not authenticated"),
    };
    let following_id = path.into_inner();

    let result = sqlx::query(
        "DELETE FROM follows WHERE follower_id = ? AND following_id = ?",
    )
    .bind(follower_id)
    .bind(following_id)
    .execute(&state.db)
    .await;

    match result {
        Ok(_) => HttpResponse::Ok().json("Unfollowed"),
        Err(_) => HttpResponse::InternalServerError().json("Failed to unfollow"),
    }
}

pub async fn get_profile(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<i64>,
) -> impl Responder {
    let current_user_id = auth::extract_user_id(&req, &state).unwrap_or(-1);
    let target_user_id = path.into_inner();

    let row = sqlx::query(
        r#"SELECT u.id, u.username, u.bio,
            (SELECT COUNT(*) FROM follows WHERE following_id = u.id) as followers_count,
            (SELECT COUNT(*) FROM follows WHERE follower_id = u.id) as following_count,
            (SELECT COUNT(*) FROM posts WHERE user_id = u.id) as posts_count,
            (SELECT COUNT(*) FROM follows WHERE follower_id = ? AND following_id = u.id) as is_following
           FROM users u
           WHERE u.id = ?"#,
    )
    .bind(current_user_id)
    .bind(target_user_id)
    .fetch_optional(&state.db)
    .await;

    match row {
        Ok(Some(r)) => {
            let profile = UserProfile {
                id: r.get("id"),
                username: r.get("username"),
                bio: r.get("bio"),
                followers_count: r.get::<i64, _>("followers_count"),
                following_count: r.get::<i64, _>("following_count"),
                posts_count: r.get::<i64, _>("posts_count"),
                is_following: r.get::<i64, _>("is_following") > 0,
            };
            HttpResponse::Ok().json(profile)
        }
        Ok(None) => HttpResponse::NotFound().json("User not found"),
        Err(_) => HttpResponse::InternalServerError().json("Failed to fetch profile"),
    }
}
