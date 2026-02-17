use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, http::header};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tweet {
    id: String,
    username: String,
    content: String,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct CreateTweet {
    username: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DeleteQuery {
    username: String,
}

struct AppState {
    tweets: Mutex<Vec<Tweet>>,
}

async fn get_tweets(data: web::Data<AppState>) -> HttpResponse {
    let tweets = data.tweets.lock().unwrap();
    let mut result: Vec<Tweet> = tweets.clone();
    result.reverse();
    HttpResponse::Ok().json(result)
}

async fn create_tweet(
    data: web::Data<AppState>,
    body: web::Json<CreateTweet>,
) -> HttpResponse {
    let username = body.username.trim().to_string();
    let content = body.content.trim().to_string();

    if username.is_empty() || content.is_empty() || content.len() > 280 {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "username must not be empty, content must be 1-280 characters"
        }));
    }

    let tweet = Tweet {
        id: Uuid::new_v4().to_string(),
        username,
        content,
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut tweets = data.tweets.lock().unwrap();
    tweets.push(tweet.clone());
    HttpResponse::Created().json(tweet)
}

async fn delete_tweet(
    data: web::Data<AppState>,
    path: web::Path<String>,
    query: web::Query<DeleteQuery>,
) -> HttpResponse {
    let tweet_id = path.into_inner();
    let mut tweets = data.tweets.lock().unwrap();

    if let Some(pos) = tweets.iter().position(|t| t.id == tweet_id) {
        if tweets[pos].username != query.username {
            return HttpResponse::Forbidden().json(serde_json::json!({
                "error": "you can only delete your own tweets"
            }));
        }
        tweets.remove(pos);
        HttpResponse::NoContent().finish()
    } else {
        HttpResponse::NotFound().json(serde_json::json!({
            "error": "tweet not found"
        }))
    }
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let data = web::Data::new(AppState {
        tweets: Mutex::new(Vec::new()),
    });

    println!("Backend running on http://127.0.0.1:8080");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:5173")
            .allowed_methods(vec!["GET", "POST", "DELETE"])
            .allowed_headers(vec![header::CONTENT_TYPE])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .route("/api/health", web::get().to(health))
            .route("/api/tweets", web::get().to(get_tweets))
            .route("/api/tweets", web::post().to(create_tweet))
            .route("/api/tweets/{id}", web::delete().to(delete_tweet))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
