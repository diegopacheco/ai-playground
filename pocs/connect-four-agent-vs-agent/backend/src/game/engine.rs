use std::sync::Arc;
use tokio::sync::RwLock;
use crate::agents::runner::AgentRunner;
use crate::game::state::GameState;
use crate::persistence::db::Database;
use crate::sse::broadcaster::{Broadcaster, GameEvent};

pub struct GameEngine {
    pub state: GameState,
    db: Arc<Database>,
    broadcaster: Arc<RwLock<Broadcaster>>,
}

impl GameEngine {
    pub fn new(state: GameState, db: Arc<Database>, broadcaster: Arc<RwLock<Broadcaster>>) -> Self {
        Self { state, db, broadcaster }
    }

    pub async fn run(&mut self) {
        self.broadcast(GameEvent::BoardUpdate {
            board: self.state.board.to_array(),
            current_player: self.state.current_agent().to_string(),
            last_move: None,
        }).await;
        while !self.state.is_game_over() {
            let agent_name = self.state.current_agent().to_string();
            let player_symbol = self.state.current_player_symbol().to_string();
            self.broadcast(GameEvent::AgentThinking {
                agent: agent_name.clone(),
            }).await;
            let prompt = self.build_prompt(&player_symbol);
            let runner = AgentRunner::new(&agent_name);
            match runner.execute(&prompt).await {
                Ok(column) => {
                    if !self.state.board.is_column_valid(column) {
                        let retry_prompt = format!(
                            "{}\n\nColumn {} is full or invalid. Choose a different column (0-6).",
                            prompt, column
                        );
                        match runner.execute(&retry_prompt).await {
                            Ok(retry_column) => {
                                if !self.state.board.is_column_valid(retry_column) {
                                    self.handle_forfeit(&agent_name).await;
                                    break;
                                }
                                self.apply_move(retry_column, &agent_name).await;
                            }
                            Err(e) => {
                                self.handle_error(&agent_name, &e).await;
                                break;
                            }
                        }
                    } else {
                        self.apply_move(column, &agent_name).await;
                    }
                }
                Err(e) => {
                    self.handle_error(&agent_name, &e).await;
                    break;
                }
            }
        }
        let duration = self.state.duration_ms().unwrap_or(0) as u64;
        self.broadcast(GameEvent::GameOver {
            winner: self.state.winner.clone(),
            is_draw: self.state.is_draw,
            duration_ms: duration,
        }).await;
        let _ = self.db.save_match(&self.state).await;
    }

    async fn apply_move(&mut self, column: u8, agent_name: &str) {
        if let Err(e) = self.state.make_move(column) {
            self.handle_error(agent_name, &e).await;
            return;
        }
        self.broadcast(GameEvent::AgentMoved {
            agent: agent_name.to_string(),
            column,
        }).await;
        self.broadcast(GameEvent::BoardUpdate {
            board: self.state.board.to_array(),
            current_player: if self.state.is_game_over() {
                "".to_string()
            } else {
                self.state.current_agent().to_string()
            },
            last_move: Some(column),
        }).await;
    }

    async fn handle_forfeit(&mut self, agent_name: &str) {
        let opponent = if agent_name == self.state.agent_a {
            self.state.agent_b.clone()
        } else {
            self.state.agent_a.clone()
        };
        self.state.winner = Some(opponent);
        self.state.ended_at = Some(chrono::Utc::now());
        self.broadcast(GameEvent::Error {
            message: format!("{} made an invalid move and forfeits", agent_name),
        }).await;
    }

    async fn handle_error(&mut self, agent_name: &str, error: &str) {
        let opponent = if agent_name == self.state.agent_a {
            self.state.agent_b.clone()
        } else {
            self.state.agent_a.clone()
        };
        self.state.winner = Some(opponent);
        self.state.ended_at = Some(chrono::Utc::now());
        self.broadcast(GameEvent::Error {
            message: format!("{} error: {}", agent_name, error),
        }).await;
    }

    fn build_prompt(&self, player_symbol: &str) -> String {
        format!(
            "You are playing Connect Four. You are player {}.\n\n\
             Current board state:\n{}\n\
             Columns are numbered 0-6 from left to right.\n\
             Choose a column to drop your piece.\n\
             Respond with ONLY a single digit (0-6) representing your chosen column.",
            player_symbol,
            self.state.board.to_text()
        )
    }

    async fn broadcast(&self, event: GameEvent) {
        let broadcaster = self.broadcaster.read().await;
        broadcaster.send(&self.state.id, event);
    }
}
