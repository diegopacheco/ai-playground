use axum::{
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;

use crate::{handlers, middleware::auth_middleware, state::AppState};

pub fn create_routes(state: Arc<AppState>) -> Router {
    let auth_routes = Router::new()
        .route("/register", post(handlers::register))
        .route("/login", post(handlers::login))
        .route("/logout", post(handlers::logout));

    let user_routes = Router::new()
        .route("/{id}", get(handlers::get_user_profile))
        .route("/{id}", put(handlers::update_user_profile))
        .route("/{id}/followers", get(handlers::get_followers))
        .route("/{id}/following", get(handlers::get_following))
        .route("/{id}/follow", post(handlers::follow_user))
        .route("/{id}/follow", delete(handlers::unfollow_user))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    let tweet_routes = Router::new()
        .route("/", post(handlers::create_tweet))
        .route("/{id}", get(handlers::get_tweet))
        .route("/{id}", delete(handlers::delete_tweet))
        .route("/feed", get(handlers::get_feed))
        .route("/user/{userId}", get(handlers::get_user_tweets))
        .route("/{id}/like", post(handlers::like_tweet))
        .route("/{id}/like", delete(handlers::unlike_tweet))
        .route("/{id}/retweet", post(handlers::retweet_tweet))
        .route("/{id}/retweet", delete(handlers::unretweet_tweet))
        .route("/{id}/comments", post(handlers::create_comment))
        .route("/{id}/comments", get(handlers::get_comments))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    let comment_routes = Router::new()
        .route("/{id}", delete(handlers::delete_comment))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    Router::new()
        .nest("/api/auth", auth_routes)
        .nest("/api/users", user_routes)
        .nest("/api/tweets", tweet_routes)
        .nest("/api/comments", comment_routes)
        .with_state(state)
}
