use serde::{Deserialize, Serialize};
use crate::game::board::{Board, Cell};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    pub id: String,
    pub board: Board,
    pub current_player: Cell,
    pub agent_a: String,
    pub agent_b: String,
    pub moves: Vec<u8>,
    pub winner: Option<String>,
    pub is_draw: bool,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub ended_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl GameState {
    pub fn new(id: String, agent_a: String, agent_b: String) -> Self {
        Self {
            id,
            board: Board::new(),
            current_player: Cell::X,
            agent_a,
            agent_b,
            moves: Vec::new(),
            winner: None,
            is_draw: false,
            started_at: chrono::Utc::now(),
            ended_at: None,
        }
    }

    pub fn current_agent(&self) -> &str {
        match self.current_player {
            Cell::X => &self.agent_a,
            Cell::O => &self.agent_b,
            Cell::Empty => unreachable!(),
        }
    }

    pub fn current_player_symbol(&self) -> &str {
        match self.current_player {
            Cell::X => "X",
            Cell::O => "O",
            Cell::Empty => unreachable!(),
        }
    }

    pub fn make_move(&mut self, column: u8) -> Result<(), String> {
        self.board.drop_piece(column, self.current_player)?;
        self.moves.push(column);
        if let Some(winner_cell) = self.board.check_winner() {
            self.winner = Some(match winner_cell {
                Cell::X => self.agent_a.clone(),
                Cell::O => self.agent_b.clone(),
                Cell::Empty => unreachable!(),
            });
            self.ended_at = Some(chrono::Utc::now());
        } else if self.board.is_full() {
            self.is_draw = true;
            self.ended_at = Some(chrono::Utc::now());
        } else {
            self.current_player = match self.current_player {
                Cell::X => Cell::O,
                Cell::O => Cell::X,
                Cell::Empty => unreachable!(),
            };
        }
        Ok(())
    }

    pub fn is_game_over(&self) -> bool {
        self.winner.is_some() || self.is_draw
    }

    pub fn duration_ms(&self) -> Option<i64> {
        self.ended_at.map(|end| (end - self.started_at).num_milliseconds())
    }
}
