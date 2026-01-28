use actix_cors::Cors;
use actix_files::Files;
use actix_web::{web, App, HttpServer};
use std::path::Path;
use super::routes;
use super::sse;
use super::state::AppState;

pub async fn start_server(base_dir: &Path, agent: &str, model: &str, cycles: u32, port: u16) -> std::io::Result<()> {
    let state = AppState::new(
        base_dir.to_path_buf(),
        agent.to_string(),
        model.to_string(),
        cycles,
    );
    let ui_path = base_dir.join("ui").join("dist");
    let ui_exists = ui_path.exists();
    println!("Starting Agent Learner Web UI...");
    println!("Server: http://127.0.0.1:{}", port);
    println!("API: http://127.0.0.1:{}/api", port);
    if ui_exists {
        println!("UI: http://127.0.0.1:{}", port);
    } else {
        println!("UI not built. Run: cd ui && bun install && bun run build");
    }
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();
        let mut app = App::new()
            .wrap(cors)
            .app_data(web::Data::new(state.clone()))
            .route("/api/health", web::get().to(routes::health))
            .route("/api/projects", web::get().to(routes::list_projects))
            .route("/api/projects/{name}", web::get().to(routes::get_project))
            .route("/api/tasks", web::post().to(routes::submit_task))
            .route("/api/tasks/{id}/status", web::get().to(routes::get_task_status))
            .route("/api/tasks/{id}/events", web::get().to(sse::events_handler))
            .route("/api/config", web::get().to(routes::get_config))
            .route("/api/config", web::post().to(routes::update_config));
        if ui_exists {
            app = app.service(Files::new("/", ui_path.clone()).index_file("index.html"));
        }
        app
    })
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
