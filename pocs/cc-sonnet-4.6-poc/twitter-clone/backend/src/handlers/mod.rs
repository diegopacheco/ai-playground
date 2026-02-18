use actix_web::web;

pub mod auth;
pub mod posts;
pub mod follows;
pub mod likes;
pub mod search;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .service(
                web::scope("/auth")
                    .route("/register", web::post().to(auth::register))
                    .route("/login", web::post().to(auth::login)),
            )
            .service(
                web::scope("/posts")
                    .route("", web::post().to(posts::create_post))
                    .route("/timeline", web::get().to(posts::timeline))
                    .route("/{user_id}", web::get().to(posts::user_posts)),
            )
            .service(
                web::scope("/users")
                    .route("/{user_id}/follow", web::post().to(follows::follow_user))
                    .route("/{user_id}/unfollow", web::delete().to(follows::unfollow_user))
                    .route("/{user_id}/profile", web::get().to(follows::get_profile)),
            )
            .service(
                web::scope("/likes")
                    .route("/{post_id}", web::post().to(likes::like_post))
                    .route("/{post_id}", web::delete().to(likes::unlike_post)),
            )
            .route("/search", web::get().to(search::search)),
    );
}
