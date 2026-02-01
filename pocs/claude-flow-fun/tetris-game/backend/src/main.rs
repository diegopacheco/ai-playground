mod config;
mod game;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, Method},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use config::GameConfig;
use game::{GameManager, GameMessage, GameState, Score};

struct AppState {
    config: RwLock<GameConfig>,
    game_manager: GameManager,
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        config: RwLock::new(GameConfig::default()),
        game_manager: GameManager::new(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/config", get(get_config))
        .route("/api/config", post(update_config))
        .route("/api/scores", get(get_scores))
        .route("/api/scores", post(save_score))
        .route("/ws", get(ws_handler))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Server running on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn get_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state.config.read().await;
    Json(config.clone())
}

async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(new_config): Json<GameConfig>,
) -> impl IntoResponse {
    let mut config = state.config.write().await;
    *config = new_config;
    Json(config.clone())
}

async fn get_scores(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let scores = state.game_manager.get_scores().await;
    Json(scores)
}

async fn save_score(
    State(state): State<Arc<AppState>>,
    Json(score): Json<Score>,
) -> impl IntoResponse {
    state.game_manager.add_score(score.clone()).await;
    Json(score)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    let config = state.config.read().await;
    let init_msg = GameMessage {
        msg_type: "init".to_string(),
        payload: serde_json::to_value(&*config).unwrap(),
    };

    if sender
        .send(Message::Text(serde_json::to_string(&init_msg).unwrap()))
        .await
        .is_err()
    {
        return;
    }
    drop(config);

    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(game_msg) = serde_json::from_str::<GameMessage>(&text) {
                    match game_msg.msg_type.as_str() {
                        "state_update" => {
                            if let Ok(game_state) =
                                serde_json::from_value::<GameState>(game_msg.payload)
                            {
                                let response = GameMessage {
                                    msg_type: "state_ack".to_string(),
                                    payload: serde_json::json!({
                                        "received": true,
                                        "score": game_state.score
                                    }),
                                };
                                if sender
                                    .send(Message::Text(serde_json::to_string(&response).unwrap()))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        "get_config" => {
                            let config = state.config.read().await;
                            let response = GameMessage {
                                msg_type: "config".to_string(),
                                payload: serde_json::to_value(&*config).unwrap(),
                            };
                            if sender
                                .send(Message::Text(serde_json::to_string(&response).unwrap()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        "game_over" => {
                            if let Ok(score) = serde_json::from_value::<Score>(game_msg.payload) {
                                state.game_manager.add_score(score).await;
                                let scores = state.game_manager.get_scores().await;
                                let response = GameMessage {
                                    msg_type: "high_scores".to_string(),
                                    payload: serde_json::to_value(&scores).unwrap(),
                                };
                                if sender
                                    .send(Message::Text(serde_json::to_string(&response).unwrap()))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        "ping" => {
                            let response = GameMessage {
                                msg_type: "pong".to_string(),
                                payload: serde_json::json!({}),
                            };
                            if sender
                                .send(Message::Text(serde_json::to_string(&response).unwrap()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Err(_) => break,
            _ => {}
        }
    }
}
