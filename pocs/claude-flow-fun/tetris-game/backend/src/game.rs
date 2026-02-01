use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_creation() {
        let score = Score {
            player_name: "Player1".to_string(),
            score: 1000,
            level: 5,
            lines_cleared: 10,
            timestamp: 1234567890,
        };
        assert_eq!(score.player_name, "Player1");
        assert_eq!(score.score, 1000);
        assert_eq!(score.level, 5);
    }

    #[test]
    fn test_score_serialization() {
        let score = Score {
            player_name: "Test".to_string(),
            score: 500,
            level: 3,
            lines_cleared: 5,
            timestamp: 1000,
        };
        let json = serde_json::to_string(&score).unwrap();
        let deserialized: Score = serde_json::from_str(&json).unwrap();
        assert_eq!(score, deserialized);
    }

    #[test]
    fn test_game_state_default() {
        let state = GameState::default();
        assert_eq!(state.board.len(), 20);
        assert_eq!(state.board[0].len(), 10);
        assert_eq!(state.score, 0);
        assert_eq!(state.level, 1);
        assert!(!state.game_over);
    }

    #[test]
    fn test_piece_creation() {
        let piece = Piece {
            piece_type: "I".to_string(),
            x: 4,
            y: 0,
            rotation: 0,
        };
        assert_eq!(piece.piece_type, "I");
        assert_eq!(piece.x, 4);
        assert_eq!(piece.y, 0);
    }

    #[test]
    fn test_game_message() {
        let msg = GameMessage {
            msg_type: "state_update".to_string(),
            payload: serde_json::json!({"score": 100}),
        };
        assert_eq!(msg.msg_type, "state_update");
    }

    #[tokio::test]
    async fn test_game_manager_add_score() {
        let manager = GameManager::new();
        let score = Score {
            player_name: "Test".to_string(),
            score: 100,
            level: 1,
            lines_cleared: 1,
            timestamp: 1000,
        };
        manager.add_score(score).await;
        let scores = manager.get_scores().await;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].player_name, "Test");
    }

    #[tokio::test]
    async fn test_game_manager_scores_sorted() {
        let manager = GameManager::new();
        manager.add_score(Score {
            player_name: "Low".to_string(),
            score: 100,
            level: 1,
            lines_cleared: 1,
            timestamp: 1000,
        }).await;
        manager.add_score(Score {
            player_name: "High".to_string(),
            score: 500,
            level: 5,
            lines_cleared: 5,
            timestamp: 2000,
        }).await;
        let scores = manager.get_scores().await;
        assert_eq!(scores[0].player_name, "High");
        assert_eq!(scores[1].player_name, "Low");
    }

    #[tokio::test]
    async fn test_game_manager_max_100_scores() {
        let manager = GameManager::new();
        for i in 0..150 {
            manager.add_score(Score {
                player_name: format!("Player{}", i),
                score: i as u64,
                level: 1,
                lines_cleared: 1,
                timestamp: i as u64,
            }).await;
        }
        let scores = manager.get_scores().await;
        assert_eq!(scores.len(), 100);
        assert_eq!(scores[0].score, 149);
    }
}
