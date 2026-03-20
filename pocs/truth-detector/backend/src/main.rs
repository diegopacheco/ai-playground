mod engine;
mod github;
mod handlers;
mod llm;
mod models;
mod persistence;
mod router;
mod sse;

use crate::models::types::AppState;
use actix_cors::Cors;
use actix_web::{App, HttpServer, web};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let db_path = std::env::var("DATABASE_PATH").unwrap_or_else(|_| "./truth_detector.db".to_string());
    let conn = rusqlite::Connection::open(&db_path).expect("Failed to open database");
    persistence::db::init_db(&conn).expect("Failed to initialize database");

    let app_state = AppState {
        db: Arc::new(Mutex::new(conn)),
        channels: Arc::new(Mutex::new(HashMap::new())),
    };

    let port: u16 = std::env::var("BACKEND_PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse()
        .unwrap_or(3000);

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(cors)
            .app_data(web::Data::new(app_state.clone()))
            .configure(router::configure)
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
