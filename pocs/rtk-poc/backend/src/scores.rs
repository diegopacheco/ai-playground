use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub const MAX_SCORES: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Score {
    pub name: String,
    pub moves: u32,
    pub seconds: u32,
}

#[derive(Debug, Deserialize)]
pub struct ScoreInput {
    pub name: String,
    pub moves: u32,
    pub seconds: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ScoreError {
    EmptyName,
    ZeroMoves,
    ZeroSeconds,
}

pub fn validate(input: &ScoreInput) -> Result<Score, ScoreError> {
    let trimmed = input.name.trim();
    if trimmed.is_empty() {
        return Err(ScoreError::EmptyName);
    }
    if input.moves == 0 {
        return Err(ScoreError::ZeroMoves);
    }
    if input.seconds == 0 {
        return Err(ScoreError::ZeroSeconds);
    }
    Ok(Score {
        name: trimmed.to_string(),
        moves: input.moves,
        seconds: input.seconds,
    })
}

#[derive(Default)]
pub struct ScoreStore {
    inner: Mutex<Vec<Score>>,
}

impl ScoreStore {
    pub fn new() -> Self {
        Self { inner: Mutex::new(Vec::new()) }
    }

    pub fn add(&self, score: Score) {
        let mut guard = self.inner.lock().unwrap();
        guard.push(score);
        guard.sort_by(|a, b| a.moves.cmp(&b.moves).then(a.seconds.cmp(&b.seconds)));
        if guard.len() > MAX_SCORES {
            guard.truncate(MAX_SCORES);
        }
    }

    pub fn list(&self) -> Vec<Score> {
        self.inner.lock().unwrap().clone()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}
