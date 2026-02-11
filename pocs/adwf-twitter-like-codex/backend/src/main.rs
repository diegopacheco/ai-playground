use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

#[derive(Clone)]
struct AppState {
    db: SqlitePool,
}

#[derive(Deserialize)]
struct CreateUserRequest {
    username: String,
}

#[derive(Serialize)]
struct UserResponse {
    id: i64,
    username: String,
}

#[derive(Deserialize)]
struct FollowRequest {
    follower_id: i64,
    followee_id: i64,
}

#[derive(Deserialize)]
struct CreatePostRequest {
    user_id: i64,
    content: String,
}

#[derive(Deserialize)]
struct LikeRequest {
    user_id: i64,
}

#[derive(Serialize, sqlx::FromRow)]
struct PostResponse {
    id: i64,
    user_id: i64,
    username: String,
    content: String,
    created_at: String,
    likes: i64,
}

#[tokio::main]
async fn main() {
    let db = init_db("sqlite://../db/app.db")
        .await
        .expect("db init failed");
    let app = app(db);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001")
        .await
        .expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}

fn app(db: SqlitePool) -> Router {
    let state = AppState { db };
    Router::new()
        .route("/health", get(health))
        .route("/api/users", post(create_user))
        .route("/api/follows", post(create_follow))
        .route("/api/posts", post(create_post).get(list_posts))
        .route("/api/posts/:id/likes", post(like_post))
        .route("/api/timeline/:user_id", get(get_timeline))
        .nest_service("/", ServeDir::new("../frontend"))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

async fn init_db(url: &str) -> Result<SqlitePool, sqlx::Error> {
    if url.starts_with("sqlite://") && !url.contains(":memory:") {
        let path = url.trim_start_matches("sqlite://");
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await;
    }
    let pool = SqlitePoolOptions::new().max_connections(5).connect(url).await?;
    sqlx::query(include_str!("../migrations/001_init.sql"))
        .execute(&pool)
        .await?;
    Ok(pool)
}

async fn health() -> &'static str {
    "ok"
}

async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUserRequest>,
) -> Result<Json<UserResponse>, StatusCode> {
    if payload.username.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let id = sqlx::query("INSERT INTO users(username) VALUES (?)")
        .bind(payload.username.trim())
        .execute(&state.db)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .last_insert_rowid();
    Ok(Json(UserResponse {
        id,
        username: payload.username,
    }))
}

async fn create_follow(
    State(state): State<AppState>,
    Json(payload): Json<FollowRequest>,
) -> Result<StatusCode, StatusCode> {
    if payload.follower_id == payload.followee_id {
        return Err(StatusCode::BAD_REQUEST);
    }
    sqlx::query("INSERT INTO follows(follower_id, followee_id) VALUES (?, ?)")
        .bind(payload.follower_id)
        .bind(payload.followee_id)
        .execute(&state.db)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    Ok(StatusCode::CREATED)
}

async fn create_post(
    State(state): State<AppState>,
    Json(payload): Json<CreatePostRequest>,
) -> Result<StatusCode, StatusCode> {
    let text = payload.content.trim();
    if text.is_empty() || text.len() > 280 {
        return Err(StatusCode::BAD_REQUEST);
    }
    sqlx::query("INSERT INTO posts(user_id, content) VALUES (?, ?)")
        .bind(payload.user_id)
        .bind(text)
        .execute(&state.db)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    Ok(StatusCode::CREATED)
}

async fn like_post(
    Path(id): Path<i64>,
    State(state): State<AppState>,
    Json(payload): Json<LikeRequest>,
) -> Result<StatusCode, StatusCode> {
    sqlx::query("INSERT INTO likes(user_id, post_id) VALUES (?, ?)")
        .bind(payload.user_id)
        .bind(id)
        .execute(&state.db)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    Ok(StatusCode::CREATED)
}

async fn list_posts(State(state): State<AppState>) -> Result<Json<Vec<PostResponse>>, StatusCode> {
    let rows = sqlx::query_as::<_, PostResponse>(
        "SELECT p.id, p.user_id, u.username, p.content, p.created_at, COALESCE(COUNT(l.post_id), 0) AS likes \
         FROM posts p \
         JOIN users u ON u.id = p.user_id \
         LEFT JOIN likes l ON l.post_id = p.id \
         GROUP BY p.id, p.user_id, u.username, p.content, p.created_at \
         ORDER BY p.id DESC",
    )
    .fetch_all(&state.db)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

async fn get_timeline(
    Path(user_id): Path<i64>,
    State(state): State<AppState>,
) -> Result<Json<Vec<PostResponse>>, StatusCode> {
    let rows = sqlx::query_as::<_, PostResponse>(
        "SELECT p.id, p.user_id, u.username, p.content, p.created_at, COALESCE(COUNT(l.post_id), 0) AS likes \
         FROM posts p \
         JOIN users u ON u.id = p.user_id \
         LEFT JOIN likes l ON l.post_id = p.id \
         WHERE p.user_id = ? OR p.user_id IN (SELECT followee_id FROM follows WHERE follower_id = ?) \
         GROUP BY p.id, p.user_id, u.username, p.content, p.created_at \
         ORDER BY p.id DESC",
    )
    .bind(user_id)
    .bind(user_id)
    .fetch_all(&state.db)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::json;
    use tower::util::ServiceExt;

    async fn test_app() -> Router {
        let db = init_db("sqlite::memory:").await.expect("db");
        app(db)
    }

    #[tokio::test]
    async fn health_is_ok() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).expect("request"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn create_user_and_post_flow() {
        let app = test_app().await;

        let user_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/users")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"username": "alice"}).to_string()))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(user_res.status(), StatusCode::OK);

        let post_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/posts")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"user_id": 1, "content": "hello"}).to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(post_res.status(), StatusCode::CREATED);

        let list_res = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/posts")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(list_res.status(), StatusCode::OK);
    }
}
