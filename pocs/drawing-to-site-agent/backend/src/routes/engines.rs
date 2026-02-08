use axum::Json;
use crate::agents::get_available_engines;
use crate::persistence::models::EngineInfo;

pub async fn get_engines() -> Json<Vec<EngineInfo>> {
    let engines = get_available_engines()
        .into_iter()
        .map(|(id, name)| EngineInfo { id, name })
        .collect();
    Json(engines)
}
