use actix_web::{web, App, HttpServer};
use actix_cors::Cors;
use actix_files::Files;
use agent_drawing_action_backend::persistence::db;
use agent_drawing_action_backend::routes::{guesses, engines};
use agent_drawing_action_backend::AppState;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let pool = db::init_db().await;
    let state = web::Data::new(AppState { pool });

    println!("Server running on http://localhost:3001");

    HttpServer::new(move || {
        let cors = Cors::permissive();
        App::new()
            .wrap(cors)
            .app_data(state.clone())
            .route("/api/engines", web::get().to(engines::get_engines))
            .route("/api/guesses", web::post().to(guesses::create_guess))
            .route("/api/guesses", web::get().to(guesses::get_guesses))
            .route("/api/guesses/{id}", web::get().to(guesses::get_guess))
            .route("/api/guesses/{id}", web::delete().to(guesses::delete_guess))
            .service(Files::new("/output", "output").show_files_listing())
    })
    .bind("0.0.0.0:3001")?
    .run()
    .await
}
