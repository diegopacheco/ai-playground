use crate::handlers::{aggregation, analysis, stream};
use actix_web::web;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .route("/analyze", web::post().to(analysis::create_analysis))
            .route("/analyze/{id}", web::get().to(analysis::get_analysis))
            .route("/analyze/{id}/stream", web::get().to(stream::stream_analysis))
            .route("/users/{username}/weekly", web::get().to(aggregation::get_weekly))
            .route("/users/{username}/latest", web::get().to(analysis::get_latest))
            .route("/leaderboard", web::get().to(aggregation::get_leaderboard))
            .route("/leaderboard/week/{week_start}", web::get().to(aggregation::get_leaderboard_week))
            .route("/leaderboard/track", web::post().to(aggregation::track_user))
            .route("/leaderboard/track/{username}", web::delete().to(aggregation::untrack_user)),
    );
}
