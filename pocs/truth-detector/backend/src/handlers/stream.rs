use crate::models::types::AppState;
use crate::sse::broadcaster;
use actix_web::{HttpResponse, web};
use futures::stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

pub async fn stream_analysis(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let analysis_id = path.into_inner();

    let rx = broadcaster::subscribe(&data.channels, &analysis_id).await;
    match rx {
        Some(receiver) => {
            let stream = BroadcastStream::new(receiver)
                .filter_map(|result| async move {
                    match result {
                        Ok(msg) => Some(Ok::<_, actix_web::Error>(web::Bytes::from(msg))),
                        Err(_) => None,
                    }
                });
            HttpResponse::Ok()
                .content_type("text/event-stream")
                .insert_header(("Cache-Control", "no-cache"))
                .insert_header(("Connection", "keep-alive"))
                .streaming(stream)
        }
        None => HttpResponse::NotFound().json(serde_json::json!({"error": "analysis not found or already completed"})),
    }
}
