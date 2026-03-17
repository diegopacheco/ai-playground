mod game_routes;

use axum::Router;
use std::sync::Arc;
use crate::AppState;

pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        .merge(game_routes::routes())
}
