pub mod validation;
pub mod db;
pub mod handlers;

use actix_web::web;
use rusqlite::Connection;
use std::collections::HashMap;
use std::sync::Mutex;
use db::AppState;

pub fn create_app_state(db_path: &str) -> web::Data<AppState> {
    let conn = Connection::open(db_path).expect("Failed to open database");
    db::init_db(&conn);
    db::seed_admin(&conn);
    let state = AppState {
        db: Mutex::new(conn),
        sessions: Mutex::new(HashMap::new()),
    };
    web::Data::new(state)
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .route("/auth/register", web::post().to(handlers::register))
            .route("/auth/login", web::post().to(handlers::login))
            .route("/auth/logout", web::post().to(handlers::logout))
            .route("/auth/me", web::get().to(handlers::me))
            .route("/posts", web::post().to(handlers::create_post))
            .route("/posts", web::get().to(handlers::list_posts))
            .route("/posts/{id}", web::get().to(handlers::get_post))
            .route("/posts/{id}", web::delete().to(handlers::delete_post))
            .route("/posts/{id}/like", web::post().to(handlers::like_post))
            .route("/posts/{id}/like", web::delete().to(handlers::unlike_post))
            .route("/users/{id}/follow", web::post().to(handlers::follow_user))
            .route("/users/{id}/follow", web::delete().to(handlers::unfollow_user))
            .route("/users/{id}/followers", web::get().to(handlers::get_followers))
            .route("/users/{id}/following", web::get().to(handlers::get_following))
            .route("/users/{id}", web::get().to(handlers::get_user_profile))
            .route("/users/{id}", web::put().to(handlers::update_user_profile))
            .route("/users/{id}/posts", web::get().to(handlers::get_user_posts))
            .route("/timeline", web::get().to(handlers::get_timeline))
            .route("/search", web::get().to(handlers::search))
            .route("/hot", web::get().to(handlers::hot_topics))
    );
}
