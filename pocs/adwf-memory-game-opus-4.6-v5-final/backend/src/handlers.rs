use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use rand::seq::SliceRandom;
use sqlx::SqlitePool;

use crate::models::*;

fn internal_error() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": "Internal server error"})),
    )
        .into_response()
}

pub async fn create_player(
    State(pool): State<SqlitePool>,
    Json(req): Json<CreatePlayerRequest>,
) -> Response {
    let result = sqlx::query_scalar::<_, i64>(
        "INSERT INTO players (name) VALUES (?) RETURNING id",
    )
    .bind(&req.name)
    .fetch_one(&pool)
    .await;

    match result {
        Ok(id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"id": id, "name": req.name})),
        )
            .into_response(),
        Err(_) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "Player name already exists"})),
        )
            .into_response(),
    }
}

pub async fn get_player_stats(
    State(pool): State<SqlitePool>,
    Path(id): Path<i64>,
) -> Response {
    let player = match sqlx::query_as::<_, (i64, String)>(
        "SELECT id, name FROM players WHERE id = ?",
    )
    .bind(id)
    .fetch_optional(&pool)
    .await
    {
        Ok(p) => p,
        Err(_) => return internal_error(),
    };

    let player = match player {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Player not found"})),
            )
                .into_response()
        }
    };

    let games_played: i32 = match sqlx::query_scalar(
        "SELECT COUNT(*) FROM games WHERE player_id = ?",
    )
    .bind(id)
    .fetch_one(&pool)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let games_won: i32 = match sqlx::query_scalar(
        "SELECT COUNT(*) FROM games WHERE player_id = ? AND status = 'completed'",
    )
    .bind(id)
    .fetch_one(&pool)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let best_score: i32 = match sqlx::query_scalar(
        "SELECT COALESCE(MAX(score), 0) FROM games WHERE player_id = ? AND status = 'completed'",
    )
    .bind(id)
    .fetch_one(&pool)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let average_moves: f64 = match sqlx::query_scalar(
        "SELECT COALESCE(AVG(moves), 0.0) FROM games WHERE player_id = ?",
    )
    .bind(id)
    .fetch_one(&pool)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let stats = PlayerStats {
        id: player.0,
        name: player.1,
        games_played,
        games_won,
        best_score,
        average_moves,
    };

    (StatusCode::OK, Json(serde_json::json!(stats))).into_response()
}

fn generate_shuffled_board() -> Vec<Card> {
    let mut values: Vec<i32> = (1..=8).flat_map(|v| vec![v, v]).collect();
    let mut rng = rand::rng();
    values.shuffle(&mut rng);
    values
        .into_iter()
        .map(|value| Card {
            value,
            flipped: false,
            matched: false,
        })
        .collect()
}

pub async fn create_game(
    State(pool): State<SqlitePool>,
    Json(req): Json<CreateGameRequest>,
) -> Response {
    let cards = generate_shuffled_board();

    let board_json = match serde_json::to_string(&cards) {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let id = match sqlx::query_scalar::<_, i64>(
        "INSERT INTO games (player_id, board) VALUES (?, ?) RETURNING id",
    )
    .bind(req.player_id)
    .bind(&board_json)
    .fetch_one(&pool)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let board_response: Vec<CardResponse> = cards
        .iter()
        .enumerate()
        .map(|(i, _)| CardResponse {
            position: i as i32,
            value: None,
            flipped: false,
            matched: false,
        })
        .collect();

    let response = GameResponse {
        id,
        player_id: req.player_id,
        board: board_response,
        moves: 0,
        matches_found: 0,
        total_pairs: 8,
        status: "in_progress".to_string(),
        score: 0,
    };

    (StatusCode::CREATED, Json(serde_json::json!(response))).into_response()
}

pub async fn get_game(
    State(pool): State<SqlitePool>,
    Path(id): Path<i64>,
) -> Response {
    let row = match sqlx::query_as::<_, (i64, i64, String, i32, i32, i32, String, i32)>(
        "SELECT id, player_id, board, moves, matches_found, total_pairs, status, score FROM games WHERE id = ?",
    )
    .bind(id)
    .fetch_optional(&pool)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let row = match row {
        Some(r) => r,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Game not found"})),
            )
                .into_response()
        }
    };

    let cards: Vec<Card> = match serde_json::from_str(&row.2) {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let board_response: Vec<CardResponse> = cards
        .iter()
        .enumerate()
        .map(|(i, card)| CardResponse {
            position: i as i32,
            value: if card.flipped || card.matched {
                Some(card.value)
            } else {
                None
            },
            flipped: card.flipped,
            matched: card.matched,
        })
        .collect();

    let response = GameResponse {
        id: row.0,
        player_id: row.1,
        board: board_response,
        moves: row.3,
        matches_found: row.4,
        total_pairs: row.5,
        status: row.6,
        score: row.7,
    };

    (StatusCode::OK, Json(serde_json::json!(response))).into_response()
}

pub async fn flip_card(
    State(pool): State<SqlitePool>,
    Path(id): Path<i64>,
    Json(req): Json<FlipRequest>,
) -> Response {
    let mut tx = match pool.begin().await {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    if let Err(_) = sqlx::query("PRAGMA foreign_keys = ON")
        .execute(&mut *tx)
        .await
    {
        return internal_error();
    }

    let row = match sqlx::query_as::<_, (i64, i64, String, i32, i32, i32, String, i32)>(
        "SELECT id, player_id, board, moves, matches_found, total_pairs, status, score FROM games WHERE id = ?",
    )
    .bind(id)
    .fetch_optional(&mut *tx)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let row = match row {
        Some(r) => r,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Game not found"})),
            )
                .into_response()
        }
    };

    if row.6 == "completed" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Game already completed"})),
        )
            .into_response();
    }

    let mut cards: Vec<Card> = match serde_json::from_str(&row.2) {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };
    let pos = req.position as usize;

    if pos >= cards.len() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid position"})),
        )
            .into_response();
    }

    if cards[pos].matched {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Card already matched"})),
        )
            .into_response();
    }

    if cards[pos].flipped {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Card already flipped"})),
        )
            .into_response();
    }

    let currently_flipped: Vec<usize> = cards
        .iter()
        .enumerate()
        .filter(|(_, c)| c.flipped && !c.matched)
        .map(|(i, _)| i)
        .collect();

    if currently_flipped.len() >= 2 {
        for &i in &currently_flipped {
            cards[i].flipped = false;
        }
    }

    cards[pos].flipped = true;

    let newly_flipped: Vec<usize> = cards
        .iter()
        .enumerate()
        .filter(|(_, c)| c.flipped && !c.matched)
        .map(|(i, _)| i)
        .collect();

    let mut moves = row.3;
    let mut matches_found = row.4;
    let mut status = row.6.clone();
    let mut score = row.7;
    let mut matched = None;

    if newly_flipped.len() == 2 {
        moves += 1;
        let a = newly_flipped[0];
        let b = newly_flipped[1];

        if cards[a].value == cards[b].value {
            cards[a].matched = true;
            cards[b].matched = true;
            cards[a].flipped = false;
            cards[b].flipped = false;
            matches_found += 1;
            matched = Some(true);

            if matches_found == 8 {
                status = "completed".to_string();
                score = std::cmp::max(1000 - (moves * 10), 100);
            }
        } else {
            matched = Some(false);
        }
    }

    let board_json = match serde_json::to_string(&cards) {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    if status == "completed" {
        if let Err(_) = sqlx::query(
            "UPDATE games SET board = ?, moves = ?, matches_found = ?, status = ?, score = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(&board_json)
        .bind(moves)
        .bind(matches_found)
        .bind(&status)
        .bind(score)
        .bind(id)
        .execute(&mut *tx)
        .await
        {
            return internal_error();
        }
    } else {
        if let Err(_) = sqlx::query(
            "UPDATE games SET board = ?, moves = ?, matches_found = ?, status = ?, score = ? WHERE id = ?",
        )
        .bind(&board_json)
        .bind(moves)
        .bind(matches_found)
        .bind(&status)
        .bind(score)
        .bind(id)
        .execute(&mut *tx)
        .await
        {
            return internal_error();
        }
    }

    if let Err(_) = sqlx::query("INSERT INTO flips (game_id, position) VALUES (?, ?)")
        .bind(id)
        .bind(req.position)
        .execute(&mut *tx)
        .await
    {
        return internal_error();
    }

    if let Err(_) = tx.commit().await {
        return internal_error();
    }

    let board_response: Vec<CardResponse> = cards
        .iter()
        .enumerate()
        .map(|(i, card)| CardResponse {
            position: i as i32,
            value: if card.flipped || card.matched {
                Some(card.value)
            } else {
                None
            },
            flipped: card.flipped,
            matched: card.matched,
        })
        .collect();

    let game_response = GameResponse {
        id,
        player_id: row.1,
        board: board_response,
        moves,
        matches_found,
        total_pairs: row.5,
        status,
        score,
    };

    let flip_response = FlipResponse {
        game: game_response,
        matched,
    };

    (StatusCode::OK, Json(serde_json::json!(flip_response))).into_response()
}

pub async fn get_leaderboard(
    State(pool): State<SqlitePool>,
) -> Response {
    let rows = match sqlx::query_as::<_, (String, i32, i32)>(
        "SELECT p.name, g.score, g.moves FROM games g JOIN players p ON g.player_id = p.id WHERE g.status = 'completed' ORDER BY g.score DESC LIMIT 10",
    )
    .fetch_all(&pool)
    .await
    {
        Ok(v) => v,
        Err(_) => return internal_error(),
    };

    let entries: Vec<LeaderboardEntry> = rows
        .into_iter()
        .map(|(player_name, score, moves)| LeaderboardEntry {
            player_name,
            score,
            moves,
        })
        .collect();

    (StatusCode::OK, Json(serde_json::json!(entries))).into_response()
}

pub fn calculate_score(moves: i32) -> i32 {
    std::cmp::max(1000 - (moves * 10), 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_shuffled_board_has_16_cards() {
        let board = generate_shuffled_board();
        assert_eq!(board.len(), 16);
    }

    #[test]
    fn test_generate_shuffled_board_has_8_pairs() {
        let board = generate_shuffled_board();
        let mut counts = std::collections::HashMap::new();
        for card in &board {
            *counts.entry(card.value).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 8);
        for (_, count) in &counts {
            assert_eq!(*count, 2);
        }
    }

    #[test]
    fn test_generate_shuffled_board_values_range_1_to_8() {
        let board = generate_shuffled_board();
        for card in &board {
            assert!(card.value >= 1 && card.value <= 8);
        }
    }

    #[test]
    fn test_generate_shuffled_board_all_cards_face_down() {
        let board = generate_shuffled_board();
        for card in &board {
            assert!(!card.flipped);
            assert!(!card.matched);
        }
    }

    #[test]
    fn test_generate_shuffled_board_is_randomized() {
        let board1 = generate_shuffled_board();
        let board2 = generate_shuffled_board();
        let values1: Vec<i32> = board1.iter().map(|c| c.value).collect();
        let values2: Vec<i32> = board2.iter().map(|c| c.value).collect();
        let mut different = false;
        for i in 0..values1.len() {
            if values1[i] != values2[i] {
                different = true;
                break;
            }
        }
        assert!(different, "Two boards should differ (extremely unlikely to be identical)");
    }

    #[test]
    fn test_calculate_score_perfect_game() {
        let score = calculate_score(8);
        assert_eq!(score, 920);
    }

    #[test]
    fn test_calculate_score_many_moves() {
        let score = calculate_score(100);
        assert_eq!(score, 100);
    }

    #[test]
    fn test_calculate_score_minimum_is_100() {
        let score = calculate_score(200);
        assert_eq!(score, 100);
    }

    #[test]
    fn test_calculate_score_zero_moves() {
        let score = calculate_score(0);
        assert_eq!(score, 1000);
    }

    #[test]
    fn test_card_serialization_roundtrip() {
        let card = Card {
            value: 5,
            flipped: true,
            matched: false,
        };
        let json = serde_json::to_string(&card).unwrap();
        let deserialized: Card = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.value, 5);
        assert!(deserialized.flipped);
        assert!(!deserialized.matched);
    }

    #[test]
    fn test_board_serialization_roundtrip() {
        let board = generate_shuffled_board();
        let json = serde_json::to_string(&board).unwrap();
        let deserialized: Vec<Card> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.len(), 16);
    }

    #[test]
    fn test_flip_logic_two_matching_cards() {
        let mut cards = vec![
            Card { value: 1, flipped: false, matched: false },
            Card { value: 1, flipped: false, matched: false },
            Card { value: 2, flipped: false, matched: false },
            Card { value: 2, flipped: false, matched: false },
        ];

        cards[0].flipped = true;
        cards[1].flipped = true;

        let newly_flipped: Vec<usize> = cards
            .iter()
            .enumerate()
            .filter(|(_, c)| c.flipped && !c.matched)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(newly_flipped.len(), 2);
        let a = newly_flipped[0];
        let b = newly_flipped[1];
        assert_eq!(cards[a].value, cards[b].value);

        cards[a].matched = true;
        cards[b].matched = true;
        cards[a].flipped = false;
        cards[b].flipped = false;

        assert!(cards[0].matched);
        assert!(cards[1].matched);
        assert!(!cards[0].flipped);
    }

    #[test]
    fn test_flip_logic_two_non_matching_cards() {
        let mut cards = vec![
            Card { value: 1, flipped: false, matched: false },
            Card { value: 2, flipped: false, matched: false },
        ];

        cards[0].flipped = true;
        cards[1].flipped = true;

        let newly_flipped: Vec<usize> = cards
            .iter()
            .enumerate()
            .filter(|(_, c)| c.flipped && !c.matched)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(newly_flipped.len(), 2);
        let a = newly_flipped[0];
        let b = newly_flipped[1];
        assert_ne!(cards[a].value, cards[b].value);
    }

    #[test]
    fn test_game_completion_detection() {
        let matches_found = 8;
        let total_pairs = 8;
        assert_eq!(matches_found, total_pairs);
        let status = if matches_found == total_pairs {
            "completed"
        } else {
            "in_progress"
        };
        assert_eq!(status, "completed");
    }

    #[test]
    fn test_game_not_complete() {
        let matches_found = 5;
        let total_pairs = 8;
        let status = if matches_found == total_pairs {
            "completed"
        } else {
            "in_progress"
        };
        assert_eq!(status, "in_progress");
    }

    #[test]
    fn test_card_response_hides_value_when_face_down() {
        let card = Card { value: 3, flipped: false, matched: false };
        let response = CardResponse {
            position: 0,
            value: if card.flipped || card.matched { Some(card.value) } else { None },
            flipped: card.flipped,
            matched: card.matched,
        };
        assert_eq!(response.value, None);
    }

    #[test]
    fn test_card_response_shows_value_when_flipped() {
        let card = Card { value: 3, flipped: true, matched: false };
        let response = CardResponse {
            position: 0,
            value: if card.flipped || card.matched { Some(card.value) } else { None },
            flipped: card.flipped,
            matched: card.matched,
        };
        assert_eq!(response.value, Some(3));
    }

    #[test]
    fn test_card_response_shows_value_when_matched() {
        let card = Card { value: 3, flipped: false, matched: true };
        let response = CardResponse {
            position: 0,
            value: if card.flipped || card.matched { Some(card.value) } else { None },
            flipped: card.flipped,
            matched: card.matched,
        };
        assert_eq!(response.value, Some(3));
    }

    #[test]
    fn test_reset_flipped_cards_when_two_already_flipped() {
        let mut cards = vec![
            Card { value: 1, flipped: true, matched: false },
            Card { value: 2, flipped: true, matched: false },
            Card { value: 3, flipped: false, matched: false },
        ];

        let currently_flipped: Vec<usize> = cards
            .iter()
            .enumerate()
            .filter(|(_, c)| c.flipped && !c.matched)
            .map(|(i, _)| i)
            .collect();

        if currently_flipped.len() >= 2 {
            for &i in &currently_flipped {
                cards[i].flipped = false;
            }
        }

        assert!(!cards[0].flipped);
        assert!(!cards[1].flipped);
    }
}
