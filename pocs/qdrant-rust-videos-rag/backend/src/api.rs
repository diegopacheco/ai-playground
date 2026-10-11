use std::sync::Arc;

use anyhow::Result;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::Config;
use crate::ollama::Ollama;
use crate::rag;
use crate::store::{COLLECTION, Store};

const DEFAULT_LIMIT: u64 = 10;
const MAX_LIMIT: u64 = 50;

#[derive(Clone)]
struct AppState {
    store: Arc<Store>,
    ollama: Ollama,
}

#[derive(Debug, PartialEq)]
pub enum ApiError {
    BadRequest(String),
    Upstream(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            ApiError::Upstream(message) => (StatusCode::BAD_GATEWAY, message),
        };
        (status, Json(json!({"error": message}))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        ApiError::Upstream(format!("{error:#}"))
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<u64>,
    pub min_minutes: Option<f64>,
    pub max_minutes: Option<f64>,
}

#[derive(Deserialize)]
struct AskRequest {
    question: String,
}

pub fn required(value: &str, field: &str) -> Result<String, ApiError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ApiError::BadRequest(format!("{field} must not be empty")));
    }
    Ok(trimmed.to_string())
}

pub fn limit(requested: Option<u64>) -> u64 {
    requested.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT)
}

pub fn minutes(min: Option<f64>, max: Option<f64>) -> Result<(), ApiError> {
    if min.is_some_and(|value| value < 0.0) || max.is_some_and(|value| value < 0.0) {
        return Err(ApiError::BadRequest("minutes must not be negative".to_string()));
    }
    if let (Some(min), Some(max)) = (min, max)
        && min > max
    {
        return Err(ApiError::BadRequest(
            "min_minutes must not be greater than max_minutes".to_string(),
        ));
    }
    Ok(())
}

async fn stats(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let videos = state.store.count().await?;
    Ok(Json(json!({"videos": videos, "collection": COLLECTION})))
}

async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<Value>, ApiError> {
    let query = required(&request.query, "query")?;
    minutes(request.min_minutes, request.max_minutes)?;
    let vector = state.ollama.embed_query(&query).await?;
    let hits = state
        .store
        .search(
            vector,
            limit(request.limit),
            request.min_minutes,
            request.max_minutes,
        )
        .await?;
    Ok(Json(json!({"hits": hits})))
}

async fn similar(
    State(state): State<AppState>,
    Path(video_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let video_id = required(&video_id, "video_id")?;
    let hits = state.store.similar(&video_id, DEFAULT_LIMIT).await?;
    Ok(Json(json!({"hits": hits})))
}

async fn ask(
    State(state): State<AppState>,
    Json(request): Json<AskRequest>,
) -> Result<Json<Value>, ApiError> {
    let question = required(&request.question, "question")?;
    let (answer, sources) = rag::ask(&state.store, &state.ollama, &question).await?;
    Ok(Json(json!({"answer": answer, "sources": sources})))
}

pub async fn serve(config: Config) -> Result<()> {
    let store = Store::connect(&config.qdrant_grpc_url)?;
    store.ensure_collection().await?;
    let state = AppState {
        store: Arc::new(store),
        ollama: Ollama::new(&config),
    };
    let app = Router::new()
        .route("/api/stats", get(stats))
        .route("/api/search", post(search))
        .route("/api/similar/{video_id}", get(similar))
        .route("/api/ask", post(ask))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", config.api_port)).await?;
    println!("listening on http://localhost:{}", config.api_port);
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_text_is_rejected_before_any_model_call_is_paid_for() {
        assert_eq!(
            required("", "query"),
            Err(ApiError::BadRequest("query must not be empty".to_string()))
        );
        assert_eq!(
            required("  \n\t ", "question"),
            Err(ApiError::BadRequest("question must not be empty".to_string()))
        );
    }

    #[test]
    fn accepted_text_is_trimmed_so_padding_does_not_change_the_embedding() {
        assert_eq!(required("  rust async  ", "query"), Ok("rust async".to_string()));
    }

    #[test]
    fn limit_is_bounded_so_one_request_cannot_dump_the_collection() {
        assert_eq!(limit(None), DEFAULT_LIMIT);
        assert_eq!(limit(Some(3)), 3);
        assert_eq!(limit(Some(0)), 1);
        assert_eq!(limit(Some(10_000)), MAX_LIMIT);
    }

    #[test]
    fn an_impossible_minutes_range_is_a_client_error_not_an_empty_result() {
        assert!(minutes(None, None).is_ok());
        assert!(minutes(Some(5.0), None).is_ok());
        assert!(minutes(Some(5.0), Some(5.0)).is_ok());
        assert!(matches!(minutes(Some(10.0), Some(5.0)), Err(ApiError::BadRequest(_))));
        assert!(matches!(minutes(Some(-1.0), None), Err(ApiError::BadRequest(_))));
    }

    #[test]
    fn errors_map_to_the_status_codes_the_ui_relies_on() {
        let bad = ApiError::BadRequest("x".to_string()).into_response();
        let upstream = ApiError::from(anyhow::anyhow!("qdrant down")).into_response();
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        assert_eq!(upstream.status(), StatusCode::BAD_GATEWAY);
    }
}
