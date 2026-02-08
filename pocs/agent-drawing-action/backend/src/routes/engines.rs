use actix_web::HttpResponse;
use crate::agents::get_available_engines;
use crate::persistence::models::EngineInfo;

pub async fn get_engines() -> HttpResponse {
    let engines: Vec<EngineInfo> = get_available_engines()
        .into_iter()
        .map(|(id, name)| EngineInfo { id, name })
        .collect();
    HttpResponse::Ok().json(engines)
}
