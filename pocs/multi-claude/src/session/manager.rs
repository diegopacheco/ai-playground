use std::path::PathBuf;
use anyhow::Result;
use crate::agent::AgentType;
use super::Session;

pub struct SessionManager {
    sessions: Vec<Session>,
    active_index: Option<usize>,
    rows: u16,
    cols: u16,
}

impl SessionManager {
    pub fn new(rows: u16, cols: u16) -> Self {
        Self {
            sessions: Vec::new(),
            active_index: None,
            rows,
            cols,
        }
    }

    pub fn create_session(&mut self, agent_type: AgentType, working_dir: PathBuf) -> Result<usize> {
        let session = Session::new(agent_type, working_dir, self.rows, self.cols)?;
        self.sessions.push(session);
        let idx = self.sessions.len() - 1;
        if self.active_index.is_none() {
            self.active_index = Some(idx);
        }
        Ok(idx)
    }

    pub fn sessions(&self) -> &[Session] {
        &self.sessions
    }

    pub fn sessions_mut(&mut self) -> &mut [Session] {
        &mut self.sessions
    }

    pub fn active_index(&self) -> Option<usize> {
        self.active_index
    }

    pub fn set_active(&mut self, index: usize) {
        if index < self.sessions.len() {
            self.active_index = Some(index);
        }
    }

    pub fn active_session(&self) -> Option<&Session> {
        self.active_index.and_then(|i| self.sessions.get(i))
    }

    pub fn active_session_mut(&mut self) -> Option<&mut Session> {
        self.active_index.and_then(|i| self.sessions.get_mut(i))
    }

    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    pub fn kill_session(&mut self, index: usize) {
        if index < self.sessions.len() {
            self.sessions[index].kill();
            self.sessions.remove(index);
            if self.sessions.is_empty() {
                self.active_index = None;
            } else if let Some(active) = self.active_index {
                if active >= self.sessions.len() {
                    self.active_index = Some(self.sessions.len() - 1);
                } else if active > index {
                    self.active_index = Some(active - 1);
                }
            }
        }
    }

    pub fn kill_all(&mut self) {
        for session in &mut self.sessions {
            session.kill();
        }
        self.sessions.clear();
        self.active_index = None;
    }

    pub fn resize_all(&mut self, rows: u16, cols: u16) {
        self.rows = rows;
        self.cols = cols;
        for session in &mut self.sessions {
            let _ = session.resize(rows, cols);
        }
    }

    pub fn poll_all(&mut self) {
        for session in &mut self.sessions {
            session.check_status();
            let _ = session.read_output();
        }
    }

    pub fn next_session(&mut self) {
        if self.sessions.is_empty() {
            return;
        }
        let next = match self.active_index {
            Some(i) => (i + 1) % self.sessions.len(),
            None => 0,
        };
        self.active_index = Some(next);
    }

    pub fn prev_session(&mut self) {
        if self.sessions.is_empty() {
            return;
        }
        let prev = match self.active_index {
            Some(0) => self.sessions.len() - 1,
            Some(i) => i - 1,
            None => 0,
        };
        self.active_index = Some(prev);
    }
}
