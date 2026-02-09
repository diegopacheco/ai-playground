use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::{json, Value};
use sqlx::PgPool;
use tower::ServiceExt;

mod common;

async fn setup_app() -> (Router, PgPool) {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/blog_platform_test".to_string());

    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("Failed to connect to test database");

    sqlx::query("DROP TABLE IF EXISTS comments CASCADE")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("DROP TABLE IF EXISTS posts CASCADE")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("DROP TABLE IF EXISTS users CASCADE")
        .execute(&pool)
        .await
        .unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS posts (
            id UUID PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            author TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS comments (
            id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            author TEXT NOT NULL,
            post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    let app = blog_platform::routes::api::router(pool.clone());
    (app, pool)
}

async fn post_json(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, value)
}

async fn get_json(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();

    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, value)
}

async fn put_json(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("PUT")
        .uri(uri)
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, value)
}

async fn delete_json(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .unwrap();

    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, value)
}

#[tokio::test]
async fn test_create_user() {
    let (app, _pool) = setup_app().await;
    let (status, body) = post_json(&app, "/api/users", json!({"name": "Alice", "email": "alice@test.com"})).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["name"], "Alice");
    assert_eq!(body["email"], "alice@test.com");
    assert!(body["id"].is_string());
}

#[tokio::test]
async fn test_list_users() {
    let (app, _pool) = setup_app().await;
    post_json(&app, "/api/users", json!({"name": "Bob", "email": "bob@test.com"})).await;
    let (status, body) = get_json(&app, "/api/users").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.as_array().unwrap().len() >= 1);
}

#[tokio::test]
async fn test_get_user_by_id() {
    let (app, _pool) = setup_app().await;
    let (_, created) = post_json(&app, "/api/users", json!({"name": "Charlie", "email": "charlie@test.com"})).await;
    let id = created["id"].as_str().unwrap();
    let (status, body) = get_json(&app, &format!("/api/users/{}", id)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["name"], "Charlie");
}

#[tokio::test]
async fn test_get_user_not_found() {
    let (app, _pool) = setup_app().await;
    let (status, body) = get_json(&app, "/api/users/00000000-0000-0000-0000-000000000000").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn test_create_post() {
    let (app, _pool) = setup_app().await;
    let (status, body) = post_json(&app, "/api/posts", json!({"title": "First Post", "content": "Hello world", "author": "Alice"})).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["title"], "First Post");
    assert_eq!(body["content"], "Hello world");
    assert_eq!(body["author"], "Alice");
}

#[tokio::test]
async fn test_list_posts() {
    let (app, _pool) = setup_app().await;
    post_json(&app, "/api/posts", json!({"title": "Post A", "content": "Content A", "author": "Alice"})).await;
    let (status, body) = get_json(&app, "/api/posts").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.as_array().unwrap().len() >= 1);
}

#[tokio::test]
async fn test_get_post_by_id() {
    let (app, _pool) = setup_app().await;
    let (_, created) = post_json(&app, "/api/posts", json!({"title": "My Post", "content": "Content", "author": "Bob"})).await;
    let id = created["id"].as_str().unwrap();
    let (status, body) = get_json(&app, &format!("/api/posts/{}", id)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["title"], "My Post");
}

#[tokio::test]
async fn test_update_post() {
    let (app, _pool) = setup_app().await;
    let (_, created) = post_json(&app, "/api/posts", json!({"title": "Old Title", "content": "Old Content", "author": "Alice"})).await;
    let id = created["id"].as_str().unwrap();
    let (status, body) = put_json(&app, &format!("/api/posts/{}", id), json!({"title": "New Title"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["title"], "New Title");
    assert_eq!(body["content"], "Old Content");
}

#[tokio::test]
async fn test_delete_post() {
    let (app, _pool) = setup_app().await;
    let (_, created) = post_json(&app, "/api/posts", json!({"title": "Delete Me", "content": "Gone", "author": "Alice"})).await;
    let id = created["id"].as_str().unwrap();
    let (status, _) = delete_json(&app, &format!("/api/posts/{}", id)).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = get_json(&app, &format!("/api/posts/{}", id)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_post_not_found() {
    let (app, _pool) = setup_app().await;
    let (status, _) = delete_json(&app, "/api/posts/00000000-0000-0000-0000-000000000000").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_create_comment() {
    let (app, _pool) = setup_app().await;
    let (_, post) = post_json(&app, "/api/posts", json!({"title": "Post", "content": "Body", "author": "Alice"})).await;
    let post_id = post["id"].as_str().unwrap();
    let (status, body) = post_json(&app, &format!("/api/posts/{}/comments", post_id), json!({"content": "Nice post!", "author": "Bob"})).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["content"], "Nice post!");
    assert_eq!(body["author"], "Bob");
}

#[tokio::test]
async fn test_list_comments() {
    let (app, _pool) = setup_app().await;
    let (_, post) = post_json(&app, "/api/posts", json!({"title": "Post", "content": "Body", "author": "Alice"})).await;
    let post_id = post["id"].as_str().unwrap();
    post_json(&app, &format!("/api/posts/{}/comments", post_id), json!({"content": "Comment 1", "author": "Bob"})).await;
    post_json(&app, &format!("/api/posts/{}/comments", post_id), json!({"content": "Comment 2", "author": "Charlie"})).await;
    let (status, body) = get_json(&app, &format!("/api/posts/{}/comments", post_id)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_delete_comment() {
    let (app, _pool) = setup_app().await;
    let (_, post) = post_json(&app, "/api/posts", json!({"title": "Post", "content": "Body", "author": "Alice"})).await;
    let post_id = post["id"].as_str().unwrap();
    let (_, comment) = post_json(&app, &format!("/api/posts/{}/comments", post_id), json!({"content": "Delete me", "author": "Bob"})).await;
    let comment_id = comment["id"].as_str().unwrap();
    let (status, _) = delete_json(&app, &format!("/api/comments/{}", comment_id)).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_create_comment_on_nonexistent_post() {
    let (app, _pool) = setup_app().await;
    let (status, body) = post_json(&app, "/api/posts/00000000-0000-0000-0000-000000000000/comments", json!({"content": "Orphan", "author": "Bob"})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn test_cascade_delete_post_removes_comments() {
    let (app, _pool) = setup_app().await;
    let (_, post) = post_json(&app, "/api/posts", json!({"title": "Post", "content": "Body", "author": "Alice"})).await;
    let post_id = post["id"].as_str().unwrap();
    post_json(&app, &format!("/api/posts/{}/comments", post_id), json!({"content": "Comment", "author": "Bob"})).await;
    delete_json(&app, &format!("/api/posts/{}", post_id)).await;
    let (status, body) = get_json(&app, &format!("/api/posts/{}/comments", post_id)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_array().unwrap().len(), 0);
}
