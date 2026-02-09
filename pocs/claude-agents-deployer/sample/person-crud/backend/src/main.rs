use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use person_crud_backend::{
    init_db, create_person, delete_person, get_admin_view, get_admin_views,
    get_person, get_persons, update_person, AppState,
};
use rusqlite::Connection;
use std::sync::Mutex;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let conn = Connection::open("persons.db").unwrap();
    init_db(&conn);
    let data = web::Data::new(AppState {
        db: Mutex::new(conn),
    });
    println!("Backend running on http://localhost:8080");
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();
        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .route("/persons", web::get().to(get_persons))
            .route("/persons/{id}", web::get().to(get_person))
            .route("/persons", web::post().to(create_person))
            .route("/persons/{id}", web::put().to(update_person))
            .route("/persons/{id}", web::delete().to(delete_person))
            .route("/admin/views", web::get().to(get_admin_views))
            .route("/admin/views/{id}", web::get().to(get_admin_view))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
