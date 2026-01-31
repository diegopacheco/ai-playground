use axum::Json;
use crate::agents::runner::list_available_agents;

pub async fn list_agents() -> Json<Vec<String>> {
    Json(list_available_agents())
}
