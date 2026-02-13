use axum::extract::{Query, State};
use axum::Json;
use sqlx::SqlitePool;

use crate::auth::AuthUser;
use crate::errors::AppError;
use crate::models::post::{PaginationParams, PostResponse};

pub async fn get_feed(
    State(pool): State<SqlitePool>,
    auth: AuthUser,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<PostResponse>>, AppError> {
    let page = params.page.unwrap_or(1).max(1);
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = (page - 1) * limit;

    let posts = sqlx::query_as::<_, FeedRow>(
        "SELECT p.id, p.user_id, p.content, p.created_at, u.username,
                (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as like_count
         FROM posts p
         INNER JOIN users u ON u.id = p.user_id
         WHERE p.user_id IN (
             SELECT following_id FROM follows WHERE follower_id = ?
         ) OR p.user_id = ?
         ORDER BY p.created_at DESC
         LIMIT ? OFFSET ?"
    )
    .bind(auth.user_id)
    .bind(auth.user_id)
    .bind(limit)
    .bind(offset)
    .fetch_all(&pool)
    .await?;

    let response: Vec<PostResponse> = posts.into_iter().map(|p| p.into_response()).collect();
    Ok(Json(response))
}

#[derive(sqlx::FromRow)]
struct FeedRow {
    id: i64,
    user_id: i64,
    content: String,
    created_at: String,
    username: String,
    like_count: i32,
}

impl FeedRow {
    fn into_response(self) -> PostResponse {
        PostResponse {
            id: self.id,
            user_id: self.user_id,
            content: self.content,
            created_at: self.created_at,
            username: self.username,
            like_count: self.like_count as i64,
        }
    }
}
