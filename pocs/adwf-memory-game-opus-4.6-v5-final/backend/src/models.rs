use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Card {
    pub value: i32,
    pub flipped: bool,
    pub matched: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateGameRequest {
    pub player_id: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FlipRequest {
    pub position: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreatePlayerRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GameResponse {
    pub id: i64,
    pub player_id: i64,
    pub board: Vec<CardResponse>,
    pub moves: i32,
    pub matches_found: i32,
    pub total_pairs: i32,
    pub status: String,
    pub score: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CardResponse {
    pub position: i32,
    pub value: Option<i32>,
    pub flipped: bool,
    pub matched: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PlayerStats {
    pub id: i64,
    pub name: String,
    pub games_played: i32,
    pub games_won: i32,
    pub best_score: i32,
    pub average_moves: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub player_name: String,
    pub score: i32,
    pub moves: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FlipResponse {
    pub game: GameResponse,
    pub matched: Option<bool>,
}
