use actix_web::{App, HttpServer};
use actix_cors::Cors;
use actix_files::Files;
use twitter_backend::{create_app_state, configure_routes};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::fs::create_dir_all("data").ok();
    std::fs::create_dir_all("uploads").ok();
    let state = create_app_state("data/twitter.db");
    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:5173")
            .allowed_methods(vec!["GET", "POST", "PUT", "DELETE"])
            .allowed_headers(vec!["Content-Type"])
            .supports_credentials()
            .max_age(3600);
        App::new()
            .wrap(cors)
            .app_data(state.clone())
            .configure(configure_routes)
            .service(Files::new("/uploads", "uploads").show_files_listing())
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
