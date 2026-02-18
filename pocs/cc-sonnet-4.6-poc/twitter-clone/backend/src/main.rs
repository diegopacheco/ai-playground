use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use sqlx::SqlitePool;

mod auth;
mod db;
mod handlers;
mod models;

pub struct AppState {
    db: SqlitePool,
    jwt_secret: String,
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:data.db".to_string());
    let jwt_secret = std::env::var("JWT_SECRET")
        .unwrap_or_else(|_| "twitter_clone_secret_key_2024".to_string());

    let pool = SqlitePool::connect(&database_url)
        .await
        .expect("Failed to connect to database");

    db::init_db(&pool).await.expect("Failed to initialize database");

    let state = web::Data::new(AppState {
        db: pool,
        jwt_secret,
    });

    println!("Backend running at http://127.0.0.1:8080");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(state.clone())
            .configure(handlers::configure_routes)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
