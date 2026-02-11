use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;
use validator::Validate;

use crate::{
    error::{AppError, Result},
    middleware::Claims,
    models::{CreateTweetRequest, Tweet, TweetResponse},
    state::AppState,
};

#[derive(Deserialize)]
pub struct FeedQuery {
    #[serde(default = "default_limit")]
    limit: i64,
    #[serde(default)]
    offset: i64,
}

fn default_limit() -> i64 {
    20
}

async fn enrich_tweet(
    state: &AppState,
    tweet: Tweet,
    current_user_id: Option<i32>,
) -> Result<TweetResponse> {
    let (author_username, author_display_name): (String, Option<String>) = sqlx::query_as(
        "SELECT username, display_name FROM users WHERE id = $1"
    )
    .bind(tweet.user_id)
    .fetch_one(&state.db)
    .await?;

    let likes_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM likes WHERE tweet_id = $1"
    )
    .bind(tweet.id)
    .fetch_one(&state.db)
    .await?;

    let retweets_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM retweets WHERE tweet_id = $1"
    )
    .bind(tweet.id)
    .fetch_one(&state.db)
    .await?;

    let comments_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM comments WHERE tweet_id = $1"
    )
    .bind(tweet.id)
    .fetch_one(&state.db)
    .await?;

    let is_liked = if let Some(user_id) = current_user_id {
        sqlx::query_scalar::<_, bool>(
            "SELECT EXISTS(SELECT 1 FROM likes WHERE user_id = $1 AND tweet_id = $2)"
        )
        .bind(user_id)
        .bind(tweet.id)
        .fetch_one(&state.db)
        .await?
    } else {
        false
    };

    let is_retweeted = if let Some(user_id) = current_user_id {
        sqlx::query_scalar::<_, bool>(
            "SELECT EXISTS(SELECT 1 FROM retweets WHERE user_id = $1 AND tweet_id = $2)"
        )
        .bind(user_id)
        .bind(tweet.id)
        .fetch_one(&state.db)
        .await?
    } else {
        false
    };

    Ok(TweetResponse {
        tweet,
        author_username,
        author_display_name,
        likes_count,
        retweets_count,
        comments_count,
        is_liked,
        is_retweeted,
    })
}

pub async fn create_tweet(
    State(state): State<Arc<AppState>>,
    Extension(claims): Extension<Claims>,
    Json(payload): Json<CreateTweetRequest>,
) -> Result<(StatusCode, Json<TweetResponse>)> {
    payload.validate().map_err(|e| AppError::BadRequest(e.to_string()))?;

    let tweet = sqlx::query_as::<_, Tweet>(
        r#"
        INSERT INTO tweets (user_id, content)
        VALUES ($1, $2)
        RETURNING id, user_id, content, created_at, updated_at
        "#,
    )
    .bind(claims.sub)
    .bind(&payload.content)
    .fetch_one(&state.db)
    .await?;

    let tweet_response = enrich_tweet(&state, tweet, Some(claims.sub)).await?;

    Ok((StatusCode::CREATED, Json(tweet_response)))
}

pub async fn get_tweet(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<TweetResponse>> {
    let current_user_id = Some(claims.sub);

    let tweet = sqlx::query_as::<_, Tweet>(
        r#"
        SELECT id, user_id, content, created_at, updated_at
        FROM tweets
        WHERE id = $1
        "#,
    )
    .bind(tweet_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Tweet not found".to_string()))?;

    let tweet_response = enrich_tweet(&state, tweet, current_user_id).await?;

    Ok(Json(tweet_response))
}

pub async fn delete_tweet(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {

    let tweet = sqlx::query_as::<_, Tweet>(
        "SELECT id, user_id, content, created_at, updated_at FROM tweets WHERE id = $1"
    )
    .bind(tweet_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Tweet not found".to_string()))?;

    if tweet.user_id != claims.sub {
        return Err(AppError::Authorization("Cannot delete another user's tweet".to_string()));
    }

    sqlx::query("DELETE FROM tweets WHERE id = $1")
        .bind(tweet_id)
        .execute(&state.db)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn get_feed(
    State(state): State<Arc<AppState>>,
    Query(params): Query<FeedQuery>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<Vec<TweetResponse>>> {

    let following_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM follows WHERE follower_id = $1"
    )
    .bind(claims.sub)
    .fetch_one(&state.db)
    .await?;

    let tweets = if following_count > 0 {
        sqlx::query_as::<_, Tweet>(
            r#"
            SELECT t.id, t.user_id, t.content, t.created_at, t.updated_at
            FROM tweets t
            INNER JOIN follows f ON f.following_id = t.user_id
            WHERE f.follower_id = $1
            ORDER BY t.created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(claims.sub)
        .bind(params.limit)
        .bind(params.offset)
        .fetch_all(&state.db)
        .await?
    } else {
        sqlx::query_as::<_, Tweet>(
            r#"
            SELECT id, user_id, content, created_at, updated_at
            FROM tweets
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(params.limit)
        .bind(params.offset)
        .fetch_all(&state.db)
        .await?
    };

    let mut tweet_responses = Vec::new();
    for tweet in tweets {
        let tweet_response = enrich_tweet(&state, tweet, Some(claims.sub)).await?;
        tweet_responses.push(tweet_response);
    }

    Ok(Json(tweet_responses))
}

pub async fn get_user_tweets(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i32>,
    Query(params): Query<FeedQuery>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<Vec<TweetResponse>>> {
    let current_user_id = Some(claims.sub);

    let tweets = sqlx::query_as::<_, Tweet>(
        r#"
        SELECT id, user_id, content, created_at, updated_at
        FROM tweets
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        "#,
    )
    .bind(user_id)
    .bind(params.limit)
    .bind(params.offset)
    .fetch_all(&state.db)
    .await?;

    let mut tweet_responses = Vec::new();
    for tweet in tweets {
        let tweet_response = enrich_tweet(&state, tweet, current_user_id).await?;
        tweet_responses.push(tweet_response);
    }

    Ok(Json(tweet_responses))
}

pub async fn like_tweet(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {
    let tweet_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM tweets WHERE id = $1)"
    )
    .bind(tweet_id)
    .fetch_one(&state.db)
    .await?;

    if !tweet_exists {
        return Err(AppError::NotFound("Tweet not found".to_string()));
    }

    sqlx::query(
        r#"
        INSERT INTO likes (user_id, tweet_id)
        VALUES ($1, $2)
        ON CONFLICT DO NOTHING
        "#,
    )
    .bind(claims.sub)
    .bind(tweet_id)
    .execute(&state.db)
    .await?;

    Ok(StatusCode::CREATED)
}

pub async fn unlike_tweet(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {
    sqlx::query(
        r#"
        DELETE FROM likes
        WHERE user_id = $1 AND tweet_id = $2
        "#,
    )
    .bind(claims.sub)
    .bind(tweet_id)
    .execute(&state.db)
    .await?;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn retweet_tweet(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {
    let tweet_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM tweets WHERE id = $1)"
    )
    .bind(tweet_id)
    .fetch_one(&state.db)
    .await?;

    if !tweet_exists {
        return Err(AppError::NotFound("Tweet not found".to_string()));
    }

    sqlx::query(
        r#"
        INSERT INTO retweets (user_id, tweet_id)
        VALUES ($1, $2)
        ON CONFLICT DO NOTHING
        "#,
    )
    .bind(claims.sub)
    .bind(tweet_id)
    .execute(&state.db)
    .await?;

    Ok(StatusCode::CREATED)
}

pub async fn unretweet_tweet(
    State(state): State<Arc<AppState>>,
    Path(tweet_id): Path<i32>,
    Extension(claims): Extension<Claims>,
) -> Result<StatusCode> {
    sqlx::query(
        r#"
        DELETE FROM retweets
        WHERE user_id = $1 AND tweet_id = $2
        "#,
    )
    .bind(claims.sub)
    .bind(tweet_id)
    .execute(&state.db)
    .await?;

    Ok(StatusCode::NO_CONTENT)
}
