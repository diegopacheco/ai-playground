use std::path::PathBuf;
use std::io::Write;
use std::sync::mpsc::Receiver;
use uuid::Uuid;
use crate::agent::{AgentType, AgentSpawner, SpawnedAgent};
use anyhow::Result;

#[allow(dead_code)]
pub struct Session {
    pub id: Uuid,
    pub agent_type: AgentType,
    pub pid: u32,
    pub working_dir: PathBuf,
    pub buffer: Vec<u8>,
    pub parser: vt100::Parser,
    output_rx: Receiver<Vec<u8>>,
    writer: Box<dyn Write + Send>,
    master: Box<dyn portable_pty::MasterPty + Send>,
    child: Box<dyn portable_pty::Child + Send + Sync>,
    pub exited: bool,
}

impl Session {
    pub fn new(agent_type: AgentType, working_dir: PathBuf, rows: u16, cols: u16) -> Result<Self> {
        let spawned = AgentSpawner::spawn(agent_type, working_dir.clone(), rows, cols)?;
        let SpawnedAgent { pid, master, output_rx, writer, child } = spawned;

        Ok(Self {
            id: Uuid::new_v4(),
            agent_type,
            pid,
            working_dir,
            buffer: Vec::with_capacity(1024 * 64),
            parser: vt100::Parser::new(rows, cols, 1000),
            output_rx,
            writer,
            master,
            child,
            exited: false,
        })
    }

    pub fn read_output(&mut self) -> Option<Vec<u8>> {
        match self.output_rx.try_recv() {
            Ok(data) => {
                self.handle_cpr_requests(&data);
                self.buffer.extend_from_slice(&data);
                self.parser.process(&data);
                Some(data)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.exited = true;
                None
            }
        }
    }

    fn handle_cpr_requests(&mut self, data: &[u8]) {
        let mut i = 0;
        while i + 3 < data.len() {
            if data[i] == 0x1b && data[i + 1] == b'[' && data[i + 2] == b'6' && data[i + 3] == b'n' {
                let _ = self.writer.write_all(b"\x1b[1;1R");
                let _ = self.writer.flush();
                i += 4;
            } else {
                i += 1;
            }
        }
    }

    pub fn write_input(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        self.writer.flush()?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn resize(&mut self, rows: u16, cols: u16) -> Result<()> {
        self.master.resize(portable_pty::PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        self.parser.set_size(rows, cols);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn screen_contents(&self) -> String {
        self.parser.screen().contents()
    }

    pub fn screen_rows(&self) -> Vec<String> {
        let screen = self.parser.screen();
        let (rows, cols) = screen.size();
        let mut result = Vec::with_capacity(rows as usize);
        for row in 0..rows {
            let mut line = String::with_capacity(cols as usize);
            for col in 0..cols {
                let cell = screen.cell(row, col).unwrap();
                line.push(cell.contents().chars().next().unwrap_or(' '));
            }
            result.push(line.trim_end().to_string());
        }
        result
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        self.exited = true;
    }

    pub fn check_status(&mut self) {
        if let Ok(Some(_)) = self.child.try_wait() {
            self.exited = true;
        }
    }
}
