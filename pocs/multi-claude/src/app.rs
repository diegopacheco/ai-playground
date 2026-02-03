use std::io;
use std::time::Duration;
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    Terminal,
};
use crate::agent::AgentType;
use crate::input::{handle_input, InputResult};
use crate::session::{SessionManager, load_layout, save_layout};
use crate::ui::{
    render_header, render_session_list, render_terminal, render_footer,
    NewSessionDialog,
};

pub enum Focus {
    SessionList,
    Terminal,
}

pub struct App {
    session_manager: SessionManager,
    dialog: NewSessionDialog,
    focus: Focus,
    list_selection: usize,
    running: bool,
}

impl App {
    pub fn new(rows: u16, cols: u16) -> Self {
        Self {
            session_manager: SessionManager::new(rows, cols),
            dialog: NewSessionDialog::new(),
            focus: Focus::SessionList,
            list_selection: 0,
            running: true,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        self.check_restore_session()?;

        while self.running {
            self.session_manager.poll_all();

            terminal.draw(|frame| {
                let size = frame.area();
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(2),
                        Constraint::Min(0),
                        Constraint::Length(2),
                    ])
                    .split(size);

                render_header(frame, chunks[0]);

                let main_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Length(20), Constraint::Min(0)])
                    .split(chunks[1]);

                render_session_list(
                    frame,
                    main_chunks[0],
                    self.session_manager.sessions(),
                    self.session_manager.active_index(),
                    matches!(self.focus, Focus::SessionList),
                );

                render_terminal(
                    frame,
                    main_chunks[1],
                    self.session_manager.active_session(),
                    matches!(self.focus, Focus::Terminal),
                );

                render_footer(frame, chunks[2], self.session_manager.count());

                self.dialog.render(frame, size);
            })?;

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key)?;
                    }
                }
            }
        }

        self.save_session()?;
        self.session_manager.kill_all();

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }

    fn handle_key(&mut self, key: event::KeyEvent) -> Result<()> {
        let result = handle_input(
            key,
            &mut self.dialog,
            matches!(self.focus, Focus::Terminal),
        );

        match result {
            InputResult::Quit => {
                self.running = false;
            }
            InputResult::NewSession => {
                self.dialog.open();
            }
            InputResult::FocusTerminal => {
                if self.session_manager.active_session().is_some() {
                    self.focus = Focus::Terminal;
                }
            }
            InputResult::SwitchSession(idx) => {
                if idx < self.session_manager.count() {
                    self.session_manager.set_active(idx);
                    self.focus = Focus::Terminal;
                }
            }
            InputResult::CreateSession => {
                let agent_type = self.dialog.selected_agent_type();
                let working_dir = self.dialog.selected_directory();
                self.dialog.close();
                if let Err(e) = self.session_manager.create_session(agent_type, working_dir) {
                    eprintln!("Failed to create session: {}", e);
                }
                self.focus = Focus::Terminal;
            }
            InputResult::NavigateUp => {
                if !self.dialog.is_open() {
                    self.session_manager.prev_session();
                }
            }
            InputResult::NavigateDown => {
                if !self.dialog.is_open() {
                    self.session_manager.next_session();
                }
            }
            InputResult::Select => {
                if !self.dialog.is_open() && self.list_selection == 0 {
                    self.dialog.open();
                } else if !self.dialog.is_open() {
                    self.focus = Focus::Terminal;
                }
            }
            InputResult::Cancel => {
                self.focus = Focus::SessionList;
            }
            InputResult::TerminalInput(data) => {
                if let Some(session) = self.session_manager.active_session_mut() {
                    let _ = session.write_input(&data);
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn check_restore_session(&mut self) -> Result<()> {
        if let Ok(Some(layout)) = load_layout() {
            if !layout.sessions.is_empty() {
                for session_data in layout.sessions {
                    let _ = self.session_manager.create_session(
                        session_data.agent_type,
                        session_data.working_dir,
                    );
                }
                if let Some(idx) = layout.active_session_index {
                    self.session_manager.set_active(idx);
                }
            }
        }
        Ok(())
    }

    fn save_session(&self) -> Result<()> {
        let sessions_owned: Vec<_> = self
            .session_manager
            .sessions()
            .iter()
            .map(|s| (s.id.to_string(), s.agent_type, s.working_dir.clone()))
            .collect();

        let sessions_ref: Vec<(&str, AgentType, &std::path::PathBuf)> = sessions_owned
            .iter()
            .map(|(id, at, wd)| (id.as_str(), *at, wd))
            .collect();

        save_layout(&sessions_ref, self.session_manager.active_index())?;
        Ok(())
    }
}
