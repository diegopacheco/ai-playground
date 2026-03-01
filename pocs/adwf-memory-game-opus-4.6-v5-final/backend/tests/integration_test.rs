use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use memory_game_backend::{create_router, db};
use serde_json::Value;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use tower::ServiceExt;

async fn setup() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    db::init_db(&pool).await;
    pool
}

async fn send(app: Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

fn post_json(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(body).unwrap()))
        .unwrap()
}

fn get_request(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

async fn create_test_player(pool: &SqlitePool, name: &str) -> Value {
    let app = create_router(pool.clone());
    let req = post_json("/api/players", &serde_json::json!({"name": name}));
    let (_, json) = send(app, req).await;
    json
}

async fn create_test_game(pool: &SqlitePool, player_id: i64) -> Value {
    let app = create_router(pool.clone());
    let req = post_json("/api/games", &serde_json::json!({"player_id": player_id}));
    let (_, json) = send(app, req).await;
    json
}

async fn flip(pool: &SqlitePool, game_id: i64, position: i32) -> (StatusCode, Value) {
    let app = create_router(pool.clone());
    let req = post_json(
        &format!("/api/games/{}/flip", game_id),
        &serde_json::json!({"position": position}),
    );
    send(app, req).await
}

async fn get_board_values(pool: &SqlitePool, game_id: i64) -> Vec<i32> {
    let row: (String,) = sqlx::query_as("SELECT board FROM games WHERE id = ?")
        .bind(game_id)
        .fetch_one(pool)
        .await
        .unwrap();
    let cards: Vec<Value> = serde_json::from_str(&row.0).unwrap();
    cards.iter().map(|c| c["value"].as_i64().unwrap() as i32).collect()
}

async fn find_pairs(pool: &SqlitePool, game_id: i64) -> Vec<(usize, usize)> {
    let values = get_board_values(pool, game_id).await;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut used: Vec<bool> = vec![false; 16];
    for i in 0..16 {
        if used[i] {
            continue;
        }
        for j in (i + 1)..16 {
            if !used[j] && values[i] == values[j] {
                pairs.push((i, j));
                used[i] = true;
                used[j] = true;
                break;
            }
        }
    }
    pairs
}

async fn play_full_game(pool: &SqlitePool, game_id: i64) {
    let pairs = find_pairs(pool, game_id).await;
    for &(a, b) in &pairs {
        flip(pool, game_id, a as i32).await;
        flip(pool, game_id, b as i32).await;
    }
}

#[tokio::test]
async fn test_create_player() {
    let pool = setup().await;
    let app = create_router(pool.clone());
    let req = post_json("/api/players", &serde_json::json!({"name": "Alice"}));
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(json["name"], "Alice");
    assert!(json["id"].as_i64().unwrap() > 0);
}

#[tokio::test]
async fn test_create_player_duplicate() {
    let pool = setup().await;
    create_test_player(&pool, "Alice").await;

    let app = create_router(pool.clone());
    let req = post_json("/api/players", &serde_json::json!({"name": "Alice"}));
    let (status, _) = send(app, req).await;

    assert_eq!(status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_create_game() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Bob").await;
    let player_id = player["id"].as_i64().unwrap();

    let app = create_router(pool.clone());
    let req = post_json("/api/games", &serde_json::json!({"player_id": player_id}));
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(json["player_id"], player_id);
    assert_eq!(json["status"], "in_progress");
    assert_eq!(json["moves"], 0);
    assert_eq!(json["matches_found"], 0);
    assert_eq!(json["total_pairs"], 8);
    assert_eq!(json["board"].as_array().unwrap().len(), 16);

    for card in json["board"].as_array().unwrap() {
        assert_eq!(card["value"], Value::Null);
        assert_eq!(card["flipped"], false);
        assert_eq!(card["matched"], false);
    }
}

#[tokio::test]
async fn test_get_game() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Charlie").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    let app = create_router(pool.clone());
    let req = get_request(&format!("/api/games/{}", game_id));
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["id"], game_id);
    assert_eq!(json["status"], "in_progress");
}

#[tokio::test]
async fn test_get_game_not_found() {
    let pool = setup().await;
    let app = create_router(pool.clone());
    let req = get_request("/api/games/9999");
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"], "Game not found");
}

#[tokio::test]
async fn test_flip_card_first() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Dave").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    let (status, json) = flip(&pool, game_id, 0).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["matched"], Value::Null);
    assert_eq!(json["game"]["moves"], 0);
    let board = json["game"]["board"].as_array().unwrap();
    assert!(board[0]["value"].as_i64().is_some());
    assert_eq!(board[0]["flipped"], true);
}

#[tokio::test]
async fn test_flip_card_pair_match() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Eve").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    let pairs = find_pairs(&pool, game_id).await;
    let (a, b) = pairs[0];

    flip(&pool, game_id, a as i32).await;
    let (status, json) = flip(&pool, game_id, b as i32).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["matched"], true);
    assert_eq!(json["game"]["moves"], 1);
    assert_eq!(json["game"]["matches_found"], 1);
}

#[tokio::test]
async fn test_flip_card_pair_no_match() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Frank").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    let values = get_board_values(&pool, game_id).await;
    let mut pos_a = 0;
    let mut pos_b = 1;
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            if values[i] != values[j] {
                pos_a = i;
                pos_b = j;
                break;
            }
        }
        if values[pos_a] != values[pos_b] {
            break;
        }
    }

    flip(&pool, game_id, pos_a as i32).await;
    let (status, json) = flip(&pool, game_id, pos_b as i32).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["matched"], false);
    assert_eq!(json["game"]["moves"], 1);
    assert_eq!(json["game"]["matches_found"], 0);
}

#[tokio::test]
async fn test_flip_invalid_position() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Grace").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    let (status, json) = flip(&pool, game_id, 99).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"], "Invalid position");
}

#[tokio::test]
async fn test_flip_already_flipped() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Hank").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    flip(&pool, game_id, 0).await;
    let (status, json) = flip(&pool, game_id, 0).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"], "Card already flipped");
}

#[tokio::test]
async fn test_flip_already_matched() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Iris").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    let pairs = find_pairs(&pool, game_id).await;
    let (a, b) = pairs[0];

    flip(&pool, game_id, a as i32).await;
    flip(&pool, game_id, b as i32).await;

    let (status, json) = flip(&pool, game_id, a as i32).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"], "Card already matched");
}

#[tokio::test]
async fn test_full_game_flow() {
    let pool = setup().await;
    let player = create_test_player(&pool, "FullGame").await;
    let player_id = player["id"].as_i64().unwrap();
    let game = create_test_game(&pool, player_id).await;
    let game_id = game["id"].as_i64().unwrap();

    let pairs = find_pairs(&pool, game_id).await;
    assert_eq!(pairs.len(), 8);

    for (idx, &(a, b)) in pairs.iter().enumerate() {
        flip(&pool, game_id, a as i32).await;
        let (status, json) = flip(&pool, game_id, b as i32).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["matched"], true);
        assert_eq!(json["game"]["matches_found"], (idx + 1) as i64);

        if idx == 7 {
            assert_eq!(json["game"]["status"], "completed");
            assert!(json["game"]["score"].as_i64().unwrap() > 0);
        }
    }

    let app = create_router(pool.clone());
    let req = get_request(&format!("/api/games/{}", game_id));
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "completed");
    assert_eq!(json["matches_found"], 8);
}

#[tokio::test]
async fn test_leaderboard_after_game() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Leader").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    play_full_game(&pool, game_id).await;

    let app = create_router(pool.clone());
    let req = get_request("/api/leaderboard");
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    let entries = json.as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["player_name"], "Leader");
    assert!(entries[0]["score"].as_i64().unwrap() > 0);
    assert!(entries[0]["moves"].as_i64().unwrap() > 0);
}

#[tokio::test]
async fn test_player_stats() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Stats").await;
    let player_id = player["id"].as_i64().unwrap();
    let game = create_test_game(&pool, player_id).await;
    let game_id = game["id"].as_i64().unwrap();

    play_full_game(&pool, game_id).await;

    let app = create_router(pool.clone());
    let req = get_request(&format!("/api/players/{}/stats", player_id));
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["name"], "Stats");
    assert_eq!(json["games_played"], 1);
    assert_eq!(json["games_won"], 1);
    assert!(json["best_score"].as_i64().unwrap() > 0);
    assert!(json["average_moves"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn test_player_stats_not_found() {
    let pool = setup().await;
    let app = create_router(pool.clone());
    let req = get_request("/api/players/9999/stats");
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"], "Player not found");
}

#[tokio::test]
async fn test_leaderboard_empty() {
    let pool = setup().await;
    let app = create_router(pool.clone());
    let req = get_request("/api/leaderboard");
    let (status, json) = send(app, req).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_flip_on_nonexistent_game() {
    let pool = setup().await;
    let (status, json) = flip(&pool, 9999, 0).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"], "Game not found");
}

#[tokio::test]
async fn test_flip_after_game_completed() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Done").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    play_full_game(&pool, game_id).await;

    let (status, json) = flip(&pool, game_id, 0).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"], "Game already completed");
}

#[tokio::test]
async fn test_unmatched_cards_reset_on_third_flip() {
    let pool = setup().await;
    let player = create_test_player(&pool, "Reset").await;
    let game = create_test_game(&pool, player["id"].as_i64().unwrap()).await;
    let game_id = game["id"].as_i64().unwrap();

    let values = get_board_values(&pool, game_id).await;

    let mut pos_a = 0;
    let mut pos_b = 1;
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            if values[i] != values[j] {
                pos_a = i;
                pos_b = j;
                break;
            }
        }
        if values[pos_a] != values[pos_b] {
            break;
        }
    }

    flip(&pool, game_id, pos_a as i32).await;
    flip(&pool, game_id, pos_b as i32).await;

    let third_pos = (0..16)
        .find(|&p| p != pos_a && p != pos_b)
        .unwrap();

    let (status, json) = flip(&pool, game_id, third_pos as i32).await;

    assert_eq!(status, StatusCode::OK);
    let board = json["game"]["board"].as_array().unwrap();
    assert_eq!(board[pos_a]["flipped"], false);
    assert_eq!(board[pos_b]["flipped"], false);
    assert_eq!(board[third_pos]["flipped"], true);
}
