use axum::{
    Router,
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tower_http::cors::{CorsLayer, Any};
use uuid::Uuid;

#[derive(Clone, Serialize, Deserialize)]
struct Tweet {
    id: String,
    username: String,
    content: String,
    created_at: String,
    likes: u32,
}

#[derive(Deserialize)]
struct CreateTweet {
    username: String,
    content: String,
}

type AppState = Arc<Mutex<Vec<Tweet>>>;

#[tokio::main]
async fn main() {
    let state: AppState = Arc::new(Mutex::new(Vec::new()));

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<axum::http::HeaderValue>().unwrap())
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/tweets", get(list_tweets).post(create_tweet))
        .route("/api/tweets/{id}", get(get_tweet).delete(delete_tweet))
        .route("/api/tweets/{id}/like", post(like_tweet))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await.unwrap();
    println!("Server running on http://localhost:3001");
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn list_tweets(State(state): State<AppState>) -> impl IntoResponse {
    let tweets = state.lock().unwrap();
    let mut sorted: Vec<Tweet> = tweets.clone();
    sorted.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Json(sorted)
}

async fn create_tweet(
    State(state): State<AppState>,
    Json(payload): Json<CreateTweet>,
) -> impl IntoResponse {
    let tweet = Tweet {
        id: Uuid::new_v4().to_string(),
        username: payload.username,
        content: payload.content,
        created_at: chrono::Utc::now().to_rfc3339(),
        likes: 0,
    };
    let mut tweets = state.lock().unwrap();
    tweets.push(tweet.clone());
    (StatusCode::CREATED, Json(tweet))
}

async fn get_tweet(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let tweets = state.lock().unwrap();
    match tweets.iter().find(|t| t.id == id) {
        Some(tweet) => Ok(Json(tweet.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn delete_tweet(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let mut tweets = state.lock().unwrap();
    let len_before = tweets.len();
    tweets.retain(|t| t.id != id);
    if tweets.len() < len_before {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn like_tweet(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let mut tweets = state.lock().unwrap();
    match tweets.iter_mut().find(|t| t.id == id) {
        Some(tweet) => {
            tweet.likes += 1;
            Ok(Json(tweet.clone()))
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}
