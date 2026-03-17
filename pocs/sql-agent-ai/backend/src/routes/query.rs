use axum::extract::{Path, State};
use axum::response::sse::{Event, Sse};
use axum::Json;
use futures::stream::Stream;
use std::convert::Infallible;
use std::sync::Arc;


use crate::app_state::AppState;
use crate::persistence::{db, models::CreateQueryRequest};
use crate::sql_agent::engine;

#[derive(serde::Serialize)]
pub struct CreateQueryResponse {
    pub id: String,
}

pub async fn create_query(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateQueryRequest>,
) -> Json<CreateQueryResponse> {
    let id = uuid::Uuid::new_v4().to_string();
    db::create_query(&state.pool, &id, &payload.question).await;

    let _ = state.broadcaster.create_channel(&id).await;

    let state_clone = state.clone();
    let id_clone = id.clone();
    let question = payload.question.clone();
    tokio::spawn(async move {
        engine::run_query(state_clone, id_clone, question).await;
    });

    Json(CreateQueryResponse { id })
}

pub async fn stream_query(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.broadcaster.subscribe(&id).await;

    let stream = async_stream::stream! {
        if let Some(mut rx) = rx {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        let event_type = match &event {
                            crate::app_state::QueryEvent::Thinking { .. } => "thinking",
                            crate::app_state::QueryEvent::SqlGenerated { .. } => "sql_generated",
                            crate::app_state::QueryEvent::SqlError { .. } => "sql_error",
                            crate::app_state::QueryEvent::SqlFixed { .. } => "sql_fixed",
                            crate::app_state::QueryEvent::QueryResult { .. } => "query_result",
                            crate::app_state::QueryEvent::Failed { .. } => "failed",
                        };
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        yield Ok(Event::default().event(event_type).data(data));

                        if matches!(event, crate::app_state::QueryEvent::QueryResult { .. } | crate::app_state::QueryEvent::Failed { .. }) {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        }
    };

    Sse::new(stream)
}
