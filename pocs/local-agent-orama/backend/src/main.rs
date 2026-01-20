mod agents;
mod files;
mod models;
mod routes;
mod worktree;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use routes::{create_session_store, run_agents, get_status, get_files, get_file_content};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let store = create_session_store();
    println!("Starting server on http://localhost:8080");
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();
        App::new()
            .wrap(cors)
            .app_data(web::Data::new(store.clone()))
            .route("/api/run", web::post().to(run_agents))
            .route("/api/status/{session_id}", web::get().to(get_status))
            .route("/api/files/{session_id}/{agent_name}", web::get().to(get_files))
            .route("/api/file/{session_id}/{agent_name}", web::get().to(get_file_content))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
