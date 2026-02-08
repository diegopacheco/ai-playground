use axum::{
    extract::{Path, State},
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use chrono::Utc;
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;
use crate::AppState;
use crate::build::engine::BuildEngine;
use crate::persistence::db;
use crate::persistence::models::{CreateProjectRequest, CreateProjectResponse, ProjectRecord};
use crate::sse::broadcaster::BuildEvent;

pub async fn create_project(
    State(state): State<AppState>,
    Json(req): Json<CreateProjectRequest>,
) -> Json<CreateProjectResponse> {
    let id = Uuid::new_v4().to_string();

    let project = ProjectRecord {
        id: id.clone(),
        name: req.name.clone(),
        engine: req.engine.clone(),
        status: "pending".to_string(),
        created_at: Utc::now().to_rfc3339(),
        completed_at: None,
        error: None,
    };

    let _ = db::create_project(&state.pool, &project).await;
    let _ = state.broadcaster.create_channel(&id).await;

    let pool = state.pool.clone();
    let broadcaster = state.broadcaster.clone();
    let project_id = id.clone();
    let image_data = req.image.clone();
    let engine = req.engine.clone();

    tokio::spawn(async move {
        let builder = BuildEngine::new(pool, broadcaster);
        builder.run_build(project_id, engine, image_data).await;
    });

    Json(CreateProjectResponse { id })
}

pub async fn get_projects(State(state): State<AppState>) -> Json<Vec<ProjectRecord>> {
    let projects = db::get_all_projects(&state.pool).await.unwrap_or_default();
    Json(projects)
}

pub async fn get_project(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Json<Option<ProjectRecord>> {
    let project = db::get_project(&state.pool, &id).await.ok().flatten();
    Json(project)
}

pub async fn delete_project(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Json<bool> {
    let output_dir = format!("output/{}", id);
    let _ = std::fs::remove_dir_all(&output_dir);
    let result = db::delete_project(&state.pool, &id).await.is_ok();
    Json(result)
}

pub async fn project_stream(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.broadcaster.subscribe(&id).await;

    let stream = async_stream::stream! {
        if let Some(rx) = rx {
            let mut stream = BroadcastStream::new(rx);
            loop {
                use tokio_stream::StreamExt;
                match stream.next().await {
                    Some(Ok(event)) => {
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        let event_type = match &event {
                            BuildEvent::StatusUpdate { .. } => "status",
                            BuildEvent::BuildComplete { .. } => "build_complete",
                            BuildEvent::Error { .. } => "error",
                        };
                        yield Ok(Event::default().event(event_type).data(data));
                    }
                    Some(Err(_)) => continue,
                    None => break,
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
