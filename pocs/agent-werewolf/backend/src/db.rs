use rusqlite::{Connection, params};
use std::sync::Mutex;
use crate::models::*;

pub struct Database {
    pub conn: Mutex<Connection>,
}

impl Database {
    pub fn new(path: &str) -> Self {
        let conn = Connection::open(path).expect("Failed to open database");
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;").unwrap();
        let db = Database { conn: Mutex::new(conn) };
        db.init_tables();
        db
    }

    fn init_tables(&self) {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                winner TEXT,
                werewolf_agent TEXT,
                deception_score INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                ended_at TEXT
            );
            CREATE TABLE IF NOT EXISTS game_agents (
                id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(id),
                agent_name TEXT NOT NULL,
                model TEXT NOT NULL,
                role TEXT NOT NULL,
                alive INTEGER NOT NULL DEFAULT 1,
                votes_correct INTEGER DEFAULT 0,
                votes_total INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS rounds (
                id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(id),
                round_number INTEGER NOT NULL,
                phase TEXT NOT NULL,
                eliminated_agent TEXT,
                eliminated_by TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                round_id TEXT NOT NULL REFERENCES rounds(id),
                agent_name TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                target TEXT,
                raw_output TEXT,
                response_time_ms INTEGER,
                created_at TEXT NOT NULL
            );"
        ).unwrap();
    }

    pub fn create_game(&self, id: &str, werewolf: &str, created_at: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO games (id, status, werewolf_agent, created_at) VALUES (?1, 'running', ?2, ?3)",
            params![id, werewolf, created_at],
        ).unwrap();
    }

    pub fn create_agent(&self, id: &str, game_id: &str, name: &str, model: &str, role: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO game_agents (id, game_id, agent_name, model, role) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, game_id, name, model, role],
        ).unwrap();
    }

    pub fn create_round(&self, id: &str, game_id: &str, round_number: i32, phase: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO rounds (id, game_id, round_number, phase) VALUES (?1, ?2, ?3, ?4)",
            params![id, game_id, round_number, phase],
        ).unwrap();
    }

    pub fn create_message(&self, id: &str, round_id: &str, agent_name: &str, msg_type: &str, content: &str, target: Option<&str>, raw_output: Option<&str>, response_time_ms: Option<i64>, created_at: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO messages (id, round_id, agent_name, message_type, content, target, raw_output, response_time_ms, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![id, round_id, agent_name, msg_type, content, target, raw_output, response_time_ms, created_at],
        ).unwrap();
    }

    pub fn update_round_elimination(&self, round_id: &str, agent: &str, by: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE rounds SET eliminated_agent = ?1, eliminated_by = ?2 WHERE id = ?3",
            params![agent, by, round_id],
        ).unwrap();
    }

    pub fn kill_agent(&self, game_id: &str, agent_name: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE game_agents SET alive = 0 WHERE game_id = ?1 AND agent_name = ?2",
            params![game_id, agent_name],
        ).unwrap();
    }

    pub fn update_agent_votes(&self, game_id: &str, agent_name: &str, correct: bool) {
        let conn = self.conn.lock().unwrap();
        if correct {
            conn.execute(
                "UPDATE game_agents SET votes_correct = votes_correct + 1, votes_total = votes_total + 1 WHERE game_id = ?1 AND agent_name = ?2",
                params![game_id, agent_name],
            ).unwrap();
        } else {
            conn.execute(
                "UPDATE game_agents SET votes_total = votes_total + 1 WHERE game_id = ?1 AND agent_name = ?2",
                params![game_id, agent_name],
            ).unwrap();
        }
    }

    pub fn end_game(&self, game_id: &str, winner: &str, deception_score: i32, ended_at: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE games SET status = 'finished', winner = ?1, deception_score = ?2, ended_at = ?3 WHERE id = ?4",
            params![winner, deception_score, ended_at, game_id],
        ).unwrap();
    }

    pub fn get_game(&self, game_id: &str) -> Option<Game> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, status, winner, werewolf_agent, deception_score, created_at, ended_at FROM games WHERE id = ?1").unwrap();
        let game = stmt.query_row(params![game_id], |row| {
            Ok(Game {
                id: row.get(0)?,
                status: row.get(1)?,
                winner: row.get(2)?,
                werewolf_agent: row.get(3)?,
                deception_score: row.get(4)?,
                created_at: row.get(5)?,
                ended_at: row.get(6)?,
                agents: vec![],
                rounds: vec![],
            })
        }).ok()?;

        let mut game = game;
        let mut stmt = conn.prepare("SELECT id, game_id, agent_name, model, role, alive, votes_correct, votes_total FROM game_agents WHERE game_id = ?1").unwrap();
        game.agents = stmt.query_map(params![game_id], |row| {
            Ok(GameAgent {
                id: row.get(0)?,
                game_id: row.get(1)?,
                agent_name: row.get(2)?,
                model: row.get(3)?,
                role: row.get(4)?,
                alive: row.get::<_, i32>(5)? == 1,
                votes_correct: row.get(6)?,
                votes_total: row.get(7)?,
            })
        }).unwrap().filter_map(|r| r.ok()).collect();

        let mut stmt = conn.prepare("SELECT id, game_id, round_number, phase, eliminated_agent, eliminated_by FROM rounds WHERE game_id = ?1 ORDER BY round_number").unwrap();
        game.rounds = stmt.query_map(params![game_id], |row| {
            Ok(Round {
                id: row.get(0)?,
                game_id: row.get(1)?,
                round_number: row.get(2)?,
                phase: row.get(3)?,
                eliminated_agent: row.get(4)?,
                eliminated_by: row.get(5)?,
                messages: vec![],
            })
        }).unwrap().filter_map(|r| r.ok()).collect();

        for round in &mut game.rounds {
            let mut stmt = conn.prepare("SELECT id, round_id, agent_name, message_type, content, target, raw_output, response_time_ms, created_at FROM messages WHERE round_id = ?1 ORDER BY created_at").unwrap();
            round.messages = stmt.query_map(params![round.id], |row| {
                Ok(Message {
                    id: row.get(0)?,
                    round_id: row.get(1)?,
                    agent_name: row.get(2)?,
                    message_type: row.get(3)?,
                    content: row.get(4)?,
                    target: row.get(5)?,
                    raw_output: row.get(6)?,
                    response_time_ms: row.get(7)?,
                    created_at: row.get(8)?,
                })
            }).unwrap().filter_map(|r| r.ok()).collect();
        }

        Some(game)
    }

    pub fn list_games(&self) -> Vec<Game> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, status, winner, werewolf_agent, deception_score, created_at, ended_at FROM games ORDER BY created_at DESC").unwrap();
        let games: Vec<Game> = stmt.query_map([], |row| {
            Ok(Game {
                id: row.get(0)?,
                status: row.get(1)?,
                winner: row.get(2)?,
                werewolf_agent: row.get(3)?,
                deception_score: row.get(4)?,
                created_at: row.get(5)?,
                ended_at: row.get(6)?,
                agents: vec![],
                rounds: vec![],
            })
        }).unwrap().filter_map(|r| r.ok()).collect();

        let mut result = vec![];
        for mut game in games {
            let mut stmt = conn.prepare("SELECT id, game_id, agent_name, model, role, alive, votes_correct, votes_total FROM game_agents WHERE game_id = ?1").unwrap();
            game.agents = stmt.query_map(params![game.id], |row| {
                Ok(GameAgent {
                    id: row.get(0)?,
                    game_id: row.get(1)?,
                    agent_name: row.get(2)?,
                    model: row.get(3)?,
                    role: row.get(4)?,
                    alive: row.get::<_, i32>(5)? == 1,
                    votes_correct: row.get(6)?,
                    votes_total: row.get(7)?,
                })
            }).unwrap().filter_map(|r| r.ok()).collect();
            result.push(game);
        }
        result
    }
}
