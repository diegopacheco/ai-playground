use std::path::PathBuf;
use anyhow::Result;
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use super::{claude, copilot, gemini, codex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    Claude,
    Copilot,
    Gemini,
    Codex,
}

impl AgentType {
    pub fn all() -> Vec<AgentType> {
        vec![
            AgentType::Claude,
            AgentType::Copilot,
            AgentType::Gemini,
            AgentType::Codex,
        ]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AgentType::Claude => "Claude",
            AgentType::Copilot => "Copilot",
            AgentType::Gemini => "Gemini",
            AgentType::Codex => "Codex",
        }
    }

    fn build_command(&self) -> (String, Vec<String>) {
        match self {
            AgentType::Claude => claude::build_command(),
            AgentType::Copilot => copilot::build_command(),
            AgentType::Gemini => gemini::build_command(),
            AgentType::Codex => codex::build_command(),
        }
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

pub struct AgentSpawner;

impl AgentSpawner {
    pub fn spawn(
        agent_type: AgentType,
        working_dir: PathBuf,
        rows: u16,
        cols: u16,
    ) -> Result<SpawnedAgent> {
        let pty_system = native_pty_system();
        let pty_pair = pty_system.openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;

        let (cmd, args) = agent_type.build_command();
        let mut command = CommandBuilder::new(&cmd);
        command.args(&args);
        command.cwd(&working_dir);
        command.env("TERM", "dumb");

        let child = pty_pair.slave.spawn_command(command)?;
        let pid = child.process_id().unwrap_or(0);

        let mut reader = pty_pair.master.try_clone_reader()?;
        let writer = pty_pair.master.take_writer()?;

        let (tx, rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = mpsc::channel();
        
        thread::spawn(move || {
            use std::io::Read;
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if tx.send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(SpawnedAgent {
            pid,
            master: pty_pair.master,
            output_rx: rx,
            writer: Box::new(writer),
            child,
        })
    }
}

pub struct SpawnedAgent {
    pub pid: u32,
    pub master: Box<dyn portable_pty::MasterPty + Send>,
    pub output_rx: Receiver<Vec<u8>>,
    pub writer: Box<dyn std::io::Write + Send>,
    pub child: Box<dyn portable_pty::Child + Send + Sync>,
}
