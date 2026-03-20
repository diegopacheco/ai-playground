use crate::engine::analyzer;
use crate::models::types::{AnalyzeRequest, AnalyzeResponse, AppState};
use crate::persistence::db;
use crate::sse::broadcaster;
use actix_web::{HttpResponse, web};

pub async fn create_analysis(
    body: web::Json<AnalyzeRequest>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let github_user = body.github_user.clone();

    {
        let conn = data.db.lock().await;
        if let Err(e) = db::insert_analysis(&conn, &id, &github_user) {
            return HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()}));
        }
    }

    broadcaster::create_channel(&data.channels, &id).await;

    let state = data.get_ref().clone();
    let analysis_id = id.clone();
    let user = github_user.clone();
    tokio::spawn(async move {
        analyzer::run_analysis(state, analysis_id, user).await;
    });

    HttpResponse::Ok().json(AnalyzeResponse { id })
}

pub async fn get_analysis(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let id = path.into_inner();
    let conn = data.db.lock().await;
    match db::get_analysis(&conn, &id) {
        Ok(Some(analysis)) => HttpResponse::Ok().json(analysis),
        Ok(None) => HttpResponse::NotFound().json(serde_json::json!({"error": "not found"})),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn get_latest(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let username = path.into_inner();
    let conn = data.db.lock().await;
    match db::get_latest_analysis(&conn, &username) {
        Ok(Some(analysis)) => HttpResponse::Ok().json(analysis),
        Ok(None) => HttpResponse::NotFound().json(serde_json::json!({"error": "not found"})),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
}
