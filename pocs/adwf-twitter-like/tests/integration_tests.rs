use sqlx::postgres::PgPoolOptions;
use twitter_clone::{config::Config, state::AppState, routes::create_routes};
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use tower::ServiceExt;
use std::sync::Arc;
use serde_json::{json, Value};
use http_body_util::BodyExt;

async fn setup_test_state() -> Arc<AppState> {
    let config = Config {
        database_url: std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/twitter_test".to_string()),
        jwt_secret: "test-secret-key-for-integration-tests".to_string(),
    };

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&config.database_url)
        .await
        .expect("Failed to connect to test database");

    sqlx::query("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        .execute(&pool)
        .await
        .ok();

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("Failed to run migrations");

    Arc::new(AppState::new(pool, config))
}

async fn body_to_json(body: Body) -> Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn register_user(app: &axum::Router, username: &str, email: &str, password: &str) -> (String, Value) {
    let register_body = json!({
        "username": username,
        "email": email,
        "password": password
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/register")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(register_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body_json = body_to_json(response.into_body()).await;
    let token = body_json["token"].as_str().unwrap().to_string();

    (token, body_json)
}

async fn login_user(app: &axum::Router, username: &str, password: &str) -> (String, Value) {
    let login_body = json!({
        "username": username,
        "password": password
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/login")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(login_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body_json = body_to_json(response.into_body()).await;
    let token = body_json["token"].as_str().unwrap().to_string();

    (token, body_json)
}

#[tokio::test]
async fn test_authentication_flow() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token, user_data) = register_user(&app, "alice", "alice@test.com", "password123").await;

    assert!(!token.is_empty());
    assert_eq!(user_data["user"]["username"], "alice");
    assert_eq!(user_data["user"]["email"], "alice@test.com");

    let (login_token, login_user_data) = login_user(&app, "alice", "password123").await;

    assert!(!login_token.is_empty());
    assert_eq!(login_user_data["user"]["username"], "alice");

    let logout_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/logout")
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(logout_response.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_authentication_validation_errors() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let invalid_email = json!({
        "username": "bob",
        "email": "not-an-email",
        "password": "password123"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/register")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(invalid_email.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let short_password = json!({
        "username": "bob",
        "email": "bob@test.com",
        "password": "123"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/register")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(short_password.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    register_user(&app, "charlie", "charlie@test.com", "password123").await;

    let duplicate_username = json!({
        "username": "charlie",
        "email": "charlie2@test.com",
        "password": "password123"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/register")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(duplicate_username.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_user_profile_operations() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token, user_data) = register_user(&app, "david", "david@test.com", "password123").await;
    let user_id = user_data["user"]["id"].as_i64().unwrap();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/users/{}", user_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let profile = body_to_json(response.into_body()).await;
    assert_eq!(profile["username"], "david");
    assert_eq!(profile["followers_count"], 0);
    assert_eq!(profile["following_count"], 0);
    assert_eq!(profile["tweets_count"], 0);

    let update_body = json!({
        "display_name": "David Smith",
        "bio": "Software engineer and coffee lover"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(format!("/api/users/{}", user_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(update_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let updated_user = body_to_json(response.into_body()).await;
    assert_eq!(updated_user["display_name"], "David Smith");
    assert_eq!(updated_user["bio"], "Software engineer and coffee lover");
}

#[tokio::test]
async fn test_follow_unfollow_operations() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token1, user1_data) = register_user(&app, "eve", "eve@test.com", "password123").await;
    let user1_id = user1_data["user"]["id"].as_i64().unwrap();

    let (token2, user2_data) = register_user(&app, "frank", "frank@test.com", "password123").await;
    let user2_id = user2_data["user"]["id"].as_i64().unwrap();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/users/{}/follow", user2_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/users/{}", user1_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let profile = body_to_json(response.into_body()).await;
    assert_eq!(profile["following_count"], 1);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/users/{}", user2_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let profile = body_to_json(response.into_body()).await;
    assert_eq!(profile["followers_count"], 1);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/users/{}/followers", user2_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let followers: Value = body_to_json(response.into_body()).await;
    assert_eq!(followers.as_array().unwrap().len(), 1);
    assert_eq!(followers[0]["username"], "eve");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/users/{}/following", user1_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let following: Value = body_to_json(response.into_body()).await;
    assert_eq!(following.as_array().unwrap().len(), 1);
    assert_eq!(following[0]["username"], "frank");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/users/{}/follow", user2_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/users/{}", user1_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let profile = body_to_json(response.into_body()).await;
    assert_eq!(profile["following_count"], 0);
}

#[tokio::test]
async fn test_tweet_operations() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token, user_data) = register_user(&app, "grace", "grace@test.com", "password123").await;
    let user_id = user_data["user"]["id"].as_i64().unwrap();

    let tweet_body = json!({
        "content": "This is my first tweet!"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(tweet_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    let tweet = body_to_json(response.into_body()).await;
    let tweet_id = tweet["id"].as_i64().unwrap();
    assert_eq!(tweet["content"], "This is my first tweet!");
    assert_eq!(tweet["author_username"], "grace");
    assert_eq!(tweet["likes_count"], 0);
    assert_eq!(tweet["retweets_count"], 0);
    assert_eq!(tweet["comments_count"], 0);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let fetched_tweet = body_to_json(response.into_body()).await;
    assert_eq!(fetched_tweet["content"], "This is my first tweet!");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/user/{}?limit=10&offset=0", user_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let user_tweets: Value = body_to_json(response.into_body()).await;
    assert_eq!(user_tweets.as_array().unwrap().len(), 1);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_tweet_validation() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token, _) = register_user(&app, "henry", "henry@test.com", "password123").await;

    let empty_tweet = json!({
        "content": ""
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(empty_tweet.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let long_tweet = json!({
        "content": "a".repeat(281)
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(long_tweet.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_like_unlike_tweet() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token1, _) = register_user(&app, "isabel", "isabel@test.com", "password123").await;
    let (token2, _) = register_user(&app, "jack", "jack@test.com", "password123").await;

    let tweet_body = json!({
        "content": "Like this tweet!"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(tweet_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    let tweet_id = tweet["id"].as_i64().unwrap();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/tweets/{}/like", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    assert_eq!(tweet["likes_count"], 1);
    assert_eq!(tweet["is_liked"], true);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/tweets/{}/like", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    assert_eq!(tweet["likes_count"], 0);
    assert_eq!(tweet["is_liked"], false);
}

#[tokio::test]
async fn test_retweet_operations() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token1, _) = register_user(&app, "kate", "kate@test.com", "password123").await;
    let (token2, _) = register_user(&app, "leo", "leo@test.com", "password123").await;

    let tweet_body = json!({
        "content": "Retweet this!"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(tweet_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    let tweet_id = tweet["id"].as_i64().unwrap();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/tweets/{}/retweet", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    assert_eq!(tweet["retweets_count"], 1);
    assert_eq!(tweet["is_retweeted"], true);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/tweets/{}/retweet", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    assert_eq!(tweet["retweets_count"], 0);
    assert_eq!(tweet["is_retweeted"], false);
}

#[tokio::test]
async fn test_comment_operations() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token1, _) = register_user(&app, "mike", "mike@test.com", "password123").await;
    let (token2, _) = register_user(&app, "nina", "nina@test.com", "password123").await;

    let tweet_body = json!({
        "content": "Comment on this tweet!"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(tweet_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    let tweet_id = tweet["id"].as_i64().unwrap();

    let comment_body = json!({
        "content": "Great tweet!"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/tweets/{}/comments", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(comment_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    let comment = body_to_json(response.into_body()).await;
    let comment_id = comment["id"].as_i64().unwrap();
    assert_eq!(comment["content"], "Great tweet!");
    assert_eq!(comment["author_username"], "nina");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}/comments", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let comments: Value = body_to_json(response.into_body()).await;
    assert_eq!(comments.as_array().unwrap().len(), 1);
    assert_eq!(comments[0]["content"], "Great tweet!");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    assert_eq!(tweet["comments_count"], 1);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/comments/{}", comment_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}/comments", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let comments: Value = body_to_json(response.into_body()).await;
    assert_eq!(comments.as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_feed_generation() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token1, user1_data) = register_user(&app, "oscar", "oscar@test.com", "password123").await;
    let user1_id = user1_data["user"]["id"].as_i64().unwrap();

    let (token2, user2_data) = register_user(&app, "paula", "paula@test.com", "password123").await;
    let user2_id = user2_data["user"]["id"].as_i64().unwrap();

    let (token3, _) = register_user(&app, "quinn", "quinn@test.com", "password123").await;

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/users/{}/follow", user2_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    for i in 1..=3 {
        let tweet_body = json!({
            "content": format!("Tweet {} from oscar", i)
        });

        app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/tweets")
                    .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(tweet_body.to_string()))
                    .unwrap()
            )
            .await
            .unwrap();
    }

    for i in 1..=2 {
        let tweet_body = json!({
            "content": format!("Tweet {} from paula", i)
        });

        app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/tweets")
                    .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(tweet_body.to_string()))
                    .unwrap()
            )
            .await
            .unwrap();
    }

    let tweet_body = json!({
        "content": "Tweet from quinn"
    });

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token3))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(tweet_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/tweets/feed?limit=20&offset=0")
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let feed: Value = body_to_json(response.into_body()).await;
    let feed_array = feed.as_array().unwrap();

    assert!(feed_array.len() >= 5);

    let user_tweets_in_feed: Vec<_> = feed_array
        .iter()
        .filter(|t| t["user_id"].as_i64() == Some(user1_id) || t["user_id"].as_i64() == Some(user2_id))
        .collect();

    assert!(user_tweets_in_feed.len() >= 5);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/tweets/feed?limit=2&offset=0")
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let limited_feed: Value = body_to_json(response.into_body()).await;
    assert_eq!(limited_feed.as_array().unwrap().len(), 2);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/tweets/feed?limit=20&offset=2")
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let offset_feed: Value = body_to_json(response.into_body()).await;
    assert!(offset_feed.as_array().unwrap().len() >= 3);
}

#[tokio::test]
async fn test_unauthorized_access() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/tweets/feed")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/tweets/feed")
                .header(header::AUTHORIZATION, "Bearer invalid-token")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let (token, _) = register_user(&app, "rachel", "rachel@test.com", "password123").await;
    let (_, user2_data) = register_user(&app, "steve", "steve@test.com", "password123").await;
    let user2_id = user2_data["user"]["id"].as_i64().unwrap();

    let update_body = json!({
        "display_name": "Unauthorized Update"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(format!("/api/users/{}", user2_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(update_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn test_data_persistence() {
    let state = setup_test_state().await;
    let app = create_routes(state.clone());

    let (token, user_data) = register_user(&app, "tom", "tom@test.com", "password123").await;
    let user_id = user_data["user"]["id"].as_i64().unwrap();

    let tweet_body = json!({
        "content": "Persistent tweet"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(tweet_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    let tweet_id = tweet["id"].as_i64().unwrap();

    let user_from_db: (String, String) = sqlx::query_as(
        "SELECT username, email FROM users WHERE id = $1"
    )
    .bind(user_id as i32)
    .fetch_one(&state.db)
    .await
    .expect("Failed to fetch user from database");

    assert_eq!(user_from_db.0, "tom");
    assert_eq!(user_from_db.1, "tom@test.com");

    let tweet_from_db: (String,) = sqlx::query_as(
        "SELECT content FROM tweets WHERE id = $1"
    )
    .bind(tweet_id as i32)
    .fetch_one(&state.db)
    .await
    .expect("Failed to fetch tweet from database");

    assert_eq!(tweet_from_db.0, "Persistent tweet");

    let app2 = create_routes(state.clone());

    let response = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let fetched_tweet = body_to_json(response.into_body()).await;
    assert_eq!(fetched_tweet["content"], "Persistent tweet");
}

#[tokio::test]
async fn test_complex_user_interaction_scenario() {
    let state = setup_test_state().await;
    let app = create_routes(state);

    let (token1, _) = register_user(&app, "user1", "user1@test.com", "password123").await;
    let (token2, user2_data) = register_user(&app, "user2", "user2@test.com", "password123").await;
    let user2_id = user2_data["user"]["id"].as_i64().unwrap();
    let (token3, user3_data) = register_user(&app, "user3", "user3@test.com", "password123").await;
    let user3_id = user3_data["user"]["id"].as_i64().unwrap();

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/users/{}/follow", user2_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/users/{}/follow", user3_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let tweet_body = json!({
        "content": "Hello from user2!"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tweets")
                .header(header::AUTHORIZATION, format!("Bearer {}", token2))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(tweet_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    let tweet = body_to_json(response.into_body()).await;
    let tweet_id = tweet["id"].as_i64().unwrap();

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/tweets/{}/like", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/tweets/{}/like", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token3))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/tweets/{}/retweet", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token3))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let comment_body = json!({
        "content": "Nice tweet!"
    });

    app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/tweets/{}/comments", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(comment_body.to_string()))
                .unwrap()
        )
        .await
        .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/tweets/{}", tweet_id))
                .header(header::AUTHORIZATION, format!("Bearer {}", token1))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let final_tweet = body_to_json(response.into_body()).await;
    assert_eq!(final_tweet["likes_count"], 2);
    assert_eq!(final_tweet["retweets_count"], 1);
    assert_eq!(final_tweet["comments_count"], 1);
    assert_eq!(final_tweet["is_liked"], true);
    assert_eq!(final_tweet["is_retweeted"], false);
}
