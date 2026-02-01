use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub player_name: String,
    pub score: u64,
    pub level: u32,
    pub lines_cleared: u32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Vec<Vec<u8>>,
    pub current_piece: Option<Piece>,
    pub next_piece: Option<Piece>,
    pub score: u64,
    pub level: u32,
    pub lines_cleared: u32,
    pub game_over: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Piece {
    pub piece_type: String,
    pub x: i32,
    pub y: i32,
    pub rotation: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameMessage {
    pub msg_type: String,
    pub payload: serde_json::Value,
}

pub struct GameManager {
    pub scores: Arc<RwLock<Vec<Score>>>,
}

impl GameManager {
    pub fn new() -> Self {
        Self {
            scores: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn add_score(&self, score: Score) {
        let mut scores = self.scores.write().await;
        scores.push(score);
        scores.sort_by(|a, b| b.score.cmp(&a.score));
        scores.truncate(100);
    }

    pub async fn get_scores(&self) -> Vec<Score> {
        let scores = self.scores.read().await;
        scores.clone()
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            board: vec![vec![0; 10]; 20],
            current_piece: None,
            next_piece: None,
            score: 0,
            level: 1,
            lines_cleared: 0,
            game_over: false,
        }
    }
}
