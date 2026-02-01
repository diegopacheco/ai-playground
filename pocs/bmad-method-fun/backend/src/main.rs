use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderValue, Method, Request, StatusCode},
    middleware::Next,
    response::{sse::Event, sse::KeepAlive, sse::Sse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::{broadcast, RwLock};
use tokio_stream::{wrappers::BroadcastStream, StreamExt};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    background: String,
    difficulty: String,
    forced_drop_interval_sec: u64,
    board_expand_interval_sec: u64,
    max_levels: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConfigUpdate {
    background: Option<String>,
    difficulty: Option<String>,
    forced_drop_interval_sec: Option<u64>,
    board_expand_interval_sec: Option<u64>,
    max_levels: Option<u64>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimerState {
    forced_drop_seconds: u64,
    board_expand_seconds: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimerUpdate {
    forced_drop_seconds: Option<u64>,
    board_expand_seconds: Option<u64>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScoreState {
    score: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScoreUpdate {
    score: u64,
}

#[derive(Clone)]
struct AppState {
    config: Arc<RwLock<Config>>,
    updates: broadcast::Sender<Config>,
    timers: Arc<RwLock<TimerState>>,
    timer_updates: broadcast::Sender<TimerState>,
    score: Arc<RwLock<ScoreState>>,
    score_updates: broadcast::Sender<ScoreState>,
}

async fn options_ok() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn get_config(State(state): State<AppState>) -> Json<Config> {
    let config = state.config.read().await.clone();
    Json(config)
}

async fn cors_middleware(req: Request<Body>, next: Next) -> Response {
    let method = req.method().clone();
    let response = next.run(req).await;
    let mut response = response;
    let headers = response.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET,POST,OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type"),
    );
    headers.insert(header::ACCESS_CONTROL_ALLOW_CREDENTIALS, HeaderValue::from_static("false"));
    if method == Method::OPTIONS {
        *response.status_mut() = StatusCode::NO_CONTENT;
    }
    response
}

async fn update_config(
    State(state): State<AppState>,
    Json(update): Json<ConfigUpdate>,
) -> StatusCode {
    let mut config = state.config.write().await;
    if let Some(value) = update.background {
        config.background = value;
    }
    if let Some(value) = update.difficulty {
        config.difficulty = value;
    }
    if let Some(value) = update.forced_drop_interval_sec {
        config.forced_drop_interval_sec = value.max(1);
    }
    if let Some(value) = update.board_expand_interval_sec {
        config.board_expand_interval_sec = value.max(1);
    }
    if let Some(value) = update.max_levels {
        config.max_levels = value.max(1);
    }
    let _ = state.updates.send(config.clone());
    StatusCode::NO_CONTENT
}

async fn update_timers(
    State(state): State<AppState>,
    Json(update): Json<TimerUpdate>,
) -> StatusCode {
    let mut timers = state.timers.write().await;
    if let Some(value) = update.forced_drop_seconds {
        timers.forced_drop_seconds = value;
    }
    if let Some(value) = update.board_expand_seconds {
        timers.board_expand_seconds = value;
    }
    let _ = state.timer_updates.send(timers.clone());
    StatusCode::NO_CONTENT
}

async fn update_score(
    State(state): State<AppState>,
    Json(update): Json<ScoreUpdate>,
) -> StatusCode {
    let mut score = state.score.write().await;
    score.score = update.score;
    let _ = state.score_updates.send(score.clone());
    StatusCode::NO_CONTENT
}

async fn config_stream(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let initial = state.config.read().await.clone();
    let initial_payload = serde_json::to_string(&initial).unwrap();
    let receiver = state.updates.subscribe();
    let stream = tokio_stream::iter([Ok(Event::default().data(initial_payload))]).chain(
        BroadcastStream::new(receiver)
            .filter_map(|item| item.ok())
            .map(|config| {
                let payload = serde_json::to_string(&config).unwrap();
                Ok(Event::default().data(payload))
            }),
    );

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
}

async fn timer_stream(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let initial = state.timers.read().await.clone();
    let initial_payload = serde_json::to_string(&initial).unwrap();
    let receiver = state.timer_updates.subscribe();
    let stream = tokio_stream::iter([Ok(Event::default().data(initial_payload))]).chain(
        BroadcastStream::new(receiver)
            .filter_map(|item| item.ok())
            .map(|timers| {
                let payload = serde_json::to_string(&timers).unwrap();
                Ok(Event::default().data(payload))
            }),
    );

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
}

async fn score_stream(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let initial = state.score.read().await.clone();
    let initial_payload = serde_json::to_string(&initial).unwrap();
    let receiver = state.score_updates.subscribe();
    let stream = tokio_stream::iter([Ok(Event::default().data(initial_payload))]).chain(
        BroadcastStream::new(receiver)
            .filter_map(|item| item.ok())
            .map(|score| {
                let payload = serde_json::to_string(&score).unwrap();
                Ok(Event::default().data(payload))
            }),
    );

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
}

#[tokio::main]
async fn main() {
    let initial = Config {
        background: "nebula".to_string(),
        difficulty: "normal".to_string(),
        forced_drop_interval_sec: 40,
        board_expand_interval_sec: 30,
        max_levels: 10,
    };
    let initial_timers = TimerState {
        forced_drop_seconds: 40,
        board_expand_seconds: 30,
    };
    let initial_score = ScoreState { score: 0 };
    let (updates, _) = broadcast::channel(32);
    let (timer_updates, _) = broadcast::channel(32);
    let (score_updates, _) = broadcast::channel(32);
    let state = AppState {
        config: Arc::new(RwLock::new(initial)),
        updates,
        timers: Arc::new(RwLock::new(initial_timers)),
        timer_updates,
        score: Arc::new(RwLock::new(initial_score)),
        score_updates,
    };

    let cors = axum::middleware::from_fn(cors_middleware);

    let app = Router::new()
        .route("/api/config", get(get_config).post(update_config).options(options_ok))
        .route("/api/config/stream", get(config_stream).options(options_ok))
        .route("/api/timers", post(update_timers).options(options_ok))
        .route("/api/timers/stream", get(timer_stream).options(options_ok))
        .route("/api/score", post(update_score).options(options_ok))
        .route("/api/score/stream", get(score_stream).options(options_ok))
        .layer(cors)
        .with_state(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
