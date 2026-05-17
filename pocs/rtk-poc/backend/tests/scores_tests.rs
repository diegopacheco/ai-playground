use memory_backend::scores::{MAX_SCORES, Score, ScoreError, ScoreInput, ScoreStore, validate};

fn score(name: &str, moves: u32, seconds: u32) -> Score {
    Score { name: name.to_string(), moves, seconds }
}

#[test]
fn store_starts_empty() {
    assert_eq!(ScoreStore::new().len(), 0);
}

#[test]
fn store_add_one() {
    let s = ScoreStore::new();
    s.add(score("a", 10, 30));
    assert_eq!(s.len(), 1);
}

#[test]
fn store_add_multiple() {
    let s = ScoreStore::new();
    s.add(score("a", 10, 30));
    s.add(score("b", 12, 25));
    s.add(score("c", 8, 40));
    assert_eq!(s.len(), 3);
}

#[test]
fn store_sorted_by_moves_ascending() {
    let s = ScoreStore::new();
    s.add(score("a", 10, 30));
    s.add(score("b", 5, 60));
    s.add(score("c", 15, 10));
    let list = s.list();
    assert_eq!(list[0].name, "b");
    assert_eq!(list[1].name, "a");
    assert_eq!(list[2].name, "c");
}

#[test]
fn store_tie_broken_by_seconds() {
    let s = ScoreStore::new();
    s.add(score("slow", 10, 60));
    s.add(score("fast", 10, 20));
    s.add(score("mid", 10, 40));
    let list = s.list();
    assert_eq!(list[0].name, "fast");
    assert_eq!(list[1].name, "mid");
    assert_eq!(list[2].name, "slow");
}

#[test]
fn store_caps_at_max_scores() {
    let s = ScoreStore::new();
    for i in 0..(MAX_SCORES as u32 + 5) {
        s.add(score(&format!("p{i}"), 100 - i, 30));
    }
    assert_eq!(s.len(), MAX_SCORES);
}

#[test]
fn store_keeps_best_when_truncating() {
    let s = ScoreStore::new();
    for i in 0..15u32 {
        s.add(score(&format!("p{i}"), 20 + i, 30));
    }
    let list = s.list();
    assert_eq!(list.first().unwrap().moves, 20);
    assert_eq!(list.last().unwrap().moves, 29);
}

#[test]
fn validate_rejects_empty_name() {
    let input = ScoreInput { name: "".into(), moves: 5, seconds: 5 };
    assert_eq!(validate(&input), Err(ScoreError::EmptyName));
}

#[test]
fn validate_rejects_whitespace_name() {
    let input = ScoreInput { name: "   ".into(), moves: 5, seconds: 5 };
    assert_eq!(validate(&input), Err(ScoreError::EmptyName));
}

#[test]
fn validate_rejects_zero_moves() {
    let input = ScoreInput { name: "x".into(), moves: 0, seconds: 5 };
    assert_eq!(validate(&input), Err(ScoreError::ZeroMoves));
}

#[test]
fn validate_rejects_zero_seconds() {
    let input = ScoreInput { name: "x".into(), moves: 5, seconds: 0 };
    assert_eq!(validate(&input), Err(ScoreError::ZeroSeconds));
}

#[test]
fn validate_trims_name() {
    let input = ScoreInput { name: "  diego  ".into(), moves: 5, seconds: 5 };
    let parsed = validate(&input).unwrap();
    assert_eq!(parsed.name, "diego");
}

#[test]
fn validate_passes_valid_input() {
    let input = ScoreInput { name: "diego".into(), moves: 8, seconds: 42 };
    let parsed = validate(&input).unwrap();
    assert_eq!(parsed, score("diego", 8, 42));
}
