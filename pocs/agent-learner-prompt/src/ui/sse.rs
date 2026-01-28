use actix_web::{web, HttpResponse, Responder};
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use super::state::AppState;

pub async fn events_handler(
    path: web::Path<String>,
    state: web::Data<Arc<AppState>>,
) -> impl Responder {
    let task_id = path.into_inner();
    let rx = state.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(move |result| {
            match result {
                Ok(event) if event.task_id == task_id => {
                    let json = serde_json::to_string(&event).unwrap_or_default();
                    Some(Ok::<_, std::convert::Infallible>(
                        actix_web::web::Bytes::from(format!("data: {}\n\n", json))
                    ))
                }
                _ => None,
            }
        });
    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(stream)
}
