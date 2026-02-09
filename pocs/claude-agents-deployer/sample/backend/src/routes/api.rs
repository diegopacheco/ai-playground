use axum::{
    Router,
    routing::{delete, get, post},
};
use sqlx::PgPool;

use crate::handlers::{comments, posts, users};

pub fn router(pool: PgPool) -> Router {
    Router::new()
        .route("/api/users", post(users::create_user).get(users::list_users))
        .route("/api/users/{id}", get(users::get_user))
        .route("/api/posts", post(posts::create_post).get(posts::list_posts))
        .route(
            "/api/posts/{id}",
            get(posts::get_post)
                .put(posts::update_post)
                .delete(posts::delete_post),
        )
        .route(
            "/api/posts/{post_id}/comments",
            post(comments::create_comment).get(comments::list_comments),
        )
        .route("/api/comments/{id}", delete(comments::delete_comment))
        .with_state(pool)
}
