use actix_cors::Cors;
use actix_web::{get, post, web, App, HttpResponse, HttpServer};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
struct Tweet {
    id: u64,
    username: String,
    content: String,
    timestamp: String,
    likes: u64,
}

#[derive(Deserialize)]
struct CreateTweet {
    username: String,
    content: String,
}

struct AppState {
    tweets: Mutex<Vec<Tweet>>,
    next_id: Mutex<u64>,
}

#[get("/api/health")]
async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

#[get("/api/tweets")]
async fn get_tweets(data: web::Data<AppState>) -> HttpResponse {
    let tweets = data.tweets.lock().unwrap();
    let mut sorted: Vec<Tweet> = tweets.clone();
    sorted.sort_by(|a, b| b.id.cmp(&a.id));
    HttpResponse::Ok().json(sorted)
}

#[post("/api/tweets")]
async fn create_tweet(data: web::Data<AppState>, body: web::Json<CreateTweet>) -> HttpResponse {
    let mut next_id = data.next_id.lock().unwrap();
    let mut tweets = data.tweets.lock().unwrap();
    let tweet = Tweet {
        id: *next_id,
        username: body.username.clone(),
        content: body.content.clone(),
        timestamp: Utc::now().to_rfc3339(),
        likes: 0,
    };
    *next_id += 1;
    tweets.push(tweet.clone());
    HttpResponse::Created().json(tweet)
}

#[post("/api/tweets/{id}/like")]
async fn like_tweet(data: web::Data<AppState>, path: web::Path<u64>) -> HttpResponse {
    let id = path.into_inner();
    let mut tweets = data.tweets.lock().unwrap();
    if let Some(tweet) = tweets.iter_mut().find(|t| t.id == id) {
        tweet.likes += 1;
        return HttpResponse::Ok().json(tweet.clone());
    }
    HttpResponse::NotFound().json(serde_json::json!({"error": "tweet not found"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let now = Utc::now().to_rfc3339();
    let seed_tweets = vec![
        Tweet { id: 1, username: "alice".into(), content: "Hello Twitter clone!".into(), timestamp: now.clone(), likes: 3 },
        Tweet { id: 2, username: "bob".into(), content: "Rust + Actix is awesome".into(), timestamp: now.clone(), likes: 5 },
        Tweet { id: 3, username: "charlie".into(), content: "Building cool stuff today".into(), timestamp: now.clone(), likes: 1 },
    ];

    let data = web::Data::new(AppState {
        tweets: Mutex::new(seed_tweets),
        next_id: Mutex::new(4),
    });

    println!("Server running on http://0.0.0.0:8080");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:3000")
            .allowed_origin("http://localhost:5173")
            .allowed_methods(vec!["GET", "POST"])
            .allowed_headers(vec!["Content-Type"])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .service(health)
            .service(get_tweets)
            .service(create_tweet)
            .service(like_tweet)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
