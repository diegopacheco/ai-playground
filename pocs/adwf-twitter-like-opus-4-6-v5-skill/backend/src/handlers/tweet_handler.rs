use axum::{extract::{Path, State}, http::StatusCode, Extension, Json};

use crate::middleware::auth_middleware::AuthUser;
use crate::models::tweet::{CreateTweetRequest, TweetWithAuthor};
use crate::AppState;

pub async fn create_tweet(
    State(state): State<AppState>,
    Extension(auth_user): Extension<AuthUser>,
    Json(payload): Json<CreateTweetRequest>,
) -> Result<(StatusCode, Json<TweetWithAuthor>), StatusCode> {
    if !payload.is_valid() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let tweet = sqlx::query_as::<_, TweetWithAuthor>(
        "SELECT t.id, t.user_id, t.content, t.created_at, u.username, u.display_name, \
         (SELECT COUNT(*) FROM likes WHERE tweet_id = t.id) as like_count \
         FROM tweets t JOIN users u ON t.user_id = u.id \
         WHERE t.id = (INSERT INTO tweets (user_id, content) VALUES ($1, $2) RETURNING id).id"
    )
    .bind(auth_user.user_id)
    .bind(&payload.content)
    .fetch_one(&state.pool)
    .await;

    match tweet {
        Ok(t) => Ok((StatusCode::CREATED, Json(t))),
        Err(_) => {
            let tweet = sqlx::query_as::<_, TweetWithAuthor>(
                "WITH new_tweet AS (INSERT INTO tweets (user_id, content) VALUES ($1, $2) RETURNING *) \
                 SELECT nt.id, nt.user_id, nt.content, nt.created_at, u.username, u.display_name, \
                 0::bigint as like_count \
                 FROM new_tweet nt JOIN users u ON nt.user_id = u.id"
            )
            .bind(auth_user.user_id)
            .bind(&payload.content)
            .fetch_one(&state.pool)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            Ok((StatusCode::CREATED, Json(tweet)))
        }
    }
}

pub async fn get_tweet(
    State(state): State<AppState>,
    Path(id): Path<i32>,
) -> Result<Json<TweetWithAuthor>, StatusCode> {
    let tweet = sqlx::query_as::<_, TweetWithAuthor>(
        "SELECT t.id, t.user_id, t.content, t.created_at, u.username, u.display_name, \
         (SELECT COUNT(*) FROM likes WHERE tweet_id = t.id) as like_count \
         FROM tweets t JOIN users u ON t.user_id = u.id WHERE t.id = $1"
    )
    .bind(id)
    .fetch_optional(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(tweet))
}

pub async fn delete_tweet(
    State(state): State<AppState>,
    Path(id): Path<i32>,
    Extension(auth_user): Extension<AuthUser>,
) -> Result<StatusCode, StatusCode> {
    let result = sqlx::query("DELETE FROM tweets WHERE id = $1 AND user_id = $2")
        .bind(id)
        .bind(auth_user.user_id)
        .execute(&state.pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    Ok(StatusCode::NO_CONTENT)
}

pub async fn get_feed(
    State(state): State<AppState>,
    Extension(auth_user): Extension<AuthUser>,
) -> Result<Json<Vec<TweetWithAuthor>>, StatusCode> {
    let tweets = sqlx::query_as::<_, TweetWithAuthor>(
        "SELECT t.id, t.user_id, t.content, t.created_at, u.username, u.display_name, \
         (SELECT COUNT(*) FROM likes WHERE tweet_id = t.id) as like_count \
         FROM tweets t JOIN users u ON t.user_id = u.id \
         WHERE t.user_id IN (SELECT following_id FROM follows WHERE follower_id = $1) \
         OR t.user_id = $1 \
         ORDER BY t.created_at DESC LIMIT 50"
    )
    .bind(auth_user.user_id)
    .fetch_all(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(tweets))
}

pub async fn get_user_tweets(
    State(state): State<AppState>,
    Path(id): Path<i32>,
) -> Result<Json<Vec<TweetWithAuthor>>, StatusCode> {
    let tweets = sqlx::query_as::<_, TweetWithAuthor>(
        "SELECT t.id, t.user_id, t.content, t.created_at, u.username, u.display_name, \
         (SELECT COUNT(*) FROM likes WHERE tweet_id = t.id) as like_count \
         FROM tweets t JOIN users u ON t.user_id = u.id \
         WHERE t.user_id = $1 ORDER BY t.created_at DESC"
    )
    .bind(id)
    .fetch_all(&state.pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(tweets))
}
