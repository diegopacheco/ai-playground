use actix_cors::Cors;
use actix_web::{web, App, HttpServer, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScoreEntry {
    id: String,
    player_name: String,
    score: u64,
    level: u8,
    lines_cleared: u32,
    difficulty: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NewScore {
    player_name: String,
    score: u64,
    level: u8,
    lines_cleared: u32,
    difficulty: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GameConfig {
    difficulty: String,
    theme: String,
    timer_enabled: bool,
    timer_minutes: u32,
}

struct AppState {
    scores: Mutex<Vec<ScoreEntry>>,
    config: Mutex<GameConfig>,
}

async fn get_scores(data: web::Data<AppState>) -> impl Responder {
    let scores = data.scores.lock().unwrap();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.score.cmp(&a.score));
    HttpResponse::Ok().json(sorted)
}

async fn add_score(data: web::Data<AppState>, body: web::Json<NewScore>) -> impl Responder {
    let mut scores = data.scores.lock().unwrap();
    let entry = ScoreEntry {
        id: Uuid::new_v4().to_string(),
        player_name: body.player_name.clone(),
        score: body.score,
        level: body.level,
        lines_cleared: body.lines_cleared,
        difficulty: body.difficulty.clone(),
        created_at: Utc::now(),
    };
    scores.push(entry.clone());
    HttpResponse::Created().json(entry)
}

async fn get_config(data: web::Data<AppState>) -> impl Responder {
    let config = data.config.lock().unwrap();
    HttpResponse::Ok().json(config.clone())
}

async fn update_config(data: web::Data<AppState>, body: web::Json<GameConfig>) -> impl Responder {
    let mut config = data.config.lock().unwrap();
    *config = body.into_inner();
    HttpResponse::Ok().json(config.clone())
}

async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let data = web::Data::new(AppState {
        scores: Mutex::new(Vec::new()),
        config: Mutex::new(GameConfig {
            difficulty: "medium".to_string(),
            theme: "classic".to_string(),
            timer_enabled: false,
            timer_minutes: 5,
        }),
    });

    println!("Tetris backend running on http://localhost:8080");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .route("/health", web::get().to(health))
            .route("/api/scores", web::get().to(get_scores))
            .route("/api/scores", web::post().to(add_score))
            .route("/api/config", web::get().to(get_config))
            .route("/api/config", web::put().to(update_config))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
