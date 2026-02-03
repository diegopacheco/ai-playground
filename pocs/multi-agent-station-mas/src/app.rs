use std::io;
use std::time::Duration;
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyEventKind, EnableMouseCapture, DisableMouseCapture, MouseEventKind},
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

#[derive(Clone, Copy)]
pub enum Focus {
    SessionList,
    Terminal,
}

pub enum PanelMode {
    Normal,
    Full(Focus),
}

pub struct App {
    session_manager: SessionManager,
    dialog: NewSessionDialog,
    focus: Focus,
    list_selection: usize,
    running: bool,
    left_panel_width: u16,
    last_error: Option<String>,
    panel_mode: PanelMode,
    search_query: String,
    visible_sessions: Vec<usize>,
    renaming: bool,
    rename_buffer: String,
}

impl App {
    pub fn new(rows: u16, cols: u16) -> Self {
        Self {
            session_manager: SessionManager::new(rows, cols),
            dialog: NewSessionDialog::new(),
            focus: Focus::SessionList,
            list_selection: 0,
            running: true,
            left_panel_width: 20,
            last_error: None,
            panel_mode: PanelMode::Normal,
            search_query: String::new(),
            visible_sessions: Vec::new(),
            renaming: false,
            rename_buffer: String::new(),
        }
    }

    pub fn run(&mut self) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        self.check_restore_session()?;

        while self.running {
            self.refresh_visible_sessions();
            self.clamp_list_selection();
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

                match self.panel_mode {
                    PanelMode::Normal => {
                        let main_chunks = Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints([Constraint::Length(self.left_panel_width), Constraint::Min(0)])
                            .split(chunks[1]);

                        render_session_list(
                            frame,
                            main_chunks[0],
                            self.session_manager.sessions(),
                            &self.visible_sessions,
                            self.session_manager.active_index(),
                            self.list_selection,
                            matches!(self.focus, Focus::SessionList),
                            &self.search_query,
                            self.renaming,
                            &self.rename_buffer,
                        );

                        render_terminal(
                            frame,
                            main_chunks[1],
                            self.session_manager.active_session(),
                            matches!(self.focus, Focus::Terminal),
                        );
                    }
                    PanelMode::Full(Focus::SessionList) => {
                        render_session_list(
                            frame,
                            chunks[1],
                            self.session_manager.sessions(),
                            &self.visible_sessions,
                            self.session_manager.active_index(),
                            self.list_selection,
                            true,
                            &self.search_query,
                            self.renaming,
                            &self.rename_buffer,
                        );
                    }
                    PanelMode::Full(Focus::Terminal) => {
                        render_terminal(
                            frame,
                            chunks[1],
                            self.session_manager.active_session(),
                            true,
                        );
                    }
                }

                render_footer(frame, chunks[2], self.session_manager.count(), self.last_error.as_deref());

                self.dialog.render(frame, size);
            })?;

            if event::poll(Duration::from_millis(50))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        self.handle_key(key)?;
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse(mouse);
                    }
                    _ => {}
                }
            }
        }

        self.save_session()?;
        self.session_manager.kill_all();

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), DisableMouseCapture, LeaveAlternateScreen)?;
        Ok(())
    }

    fn handle_mouse(&mut self, mouse: event::MouseEvent) {
        if let MouseEventKind::Down(_) = mouse.kind {
            if mouse.row >= 2 && mouse.row < mouse.row.saturating_add(100) {
                if mouse.column < self.left_panel_width {
                    self.focus = Focus::SessionList;
                } else if self.session_manager.active_session().is_some() {
                    self.focus = Focus::Terminal;
                }
            }
        }
    }

    fn handle_key(&mut self, key: event::KeyEvent) -> Result<()> {
        if self.renaming {
            return self.handle_rename_input(key);
        }

        if matches!(self.focus, Focus::SessionList) && !self.dialog.is_open() {
            if key.modifiers.is_empty() {
                match key.code {
                    event::KeyCode::Char(c) => {
                        if c != 'q' && c != 'c' && c != 'r' {
                            self.search_query.push(c);
                            return Ok(());
                        }
                    }
                    event::KeyCode::Backspace => {
                        self.search_query.pop();
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

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
            InputResult::ToggleFullScreen => {
                self.panel_mode = match self.panel_mode {
                    PanelMode::Normal => PanelMode::Full(self.focus),
                    PanelMode::Full(_) => PanelMode::Normal,
                };
            }
            InputResult::ToggleFocus => {
                self.focus = match self.focus {
                    Focus::SessionList => {
                        if self.session_manager.active_session().is_some() {
                            Focus::Terminal
                        } else {
                            Focus::SessionList
                        }
                    }
                    Focus::Terminal => Focus::SessionList,
                };
                if let PanelMode::Full(_) = self.panel_mode {
                    self.panel_mode = PanelMode::Full(self.focus);
                }
            }
            InputResult::CreateSession => {
                let agent_type = self.dialog.selected_agent_type();
                let working_dir = self.dialog.selected_directory();
                self.dialog.close();
                match self.session_manager.create_session(agent_type, working_dir) {
                    Ok(idx) => {
                        self.list_selection = self
                            .index_in_visible(idx)
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        self.session_manager.set_active(idx);
                        self.last_error = None;
                    }
                    Err(err) => {
                        self.last_error = Some(format!("Failed to start {}: {}", agent_type, err));
                    }
                }
                self.focus = Focus::Terminal;
            }
            InputResult::KillSession => {
                if self.list_selection > 0 {
                    if let Some(session_idx) = self.selected_session_index() {
                        self.session_manager.kill_session(session_idx);
                    }
                    self.refresh_visible_sessions();
                    self.clamp_list_selection();
                    if let Some(session_idx) = self.selected_session_index() {
                        self.session_manager.set_active(session_idx);
                    }
                }
            }
            InputResult::NavigateUp => {
                if !self.dialog.is_open() {
                    if self.list_selection > 0 {
                        self.list_selection -= 1;
                        if let Some(session_idx) = self.selected_session_index() {
                            self.session_manager.set_active(session_idx);
                        }
                    }
                }
            }
            InputResult::NavigateDown => {
                if !self.dialog.is_open() {
                    let max = self.visible_sessions.len();
                    if self.list_selection < max {
                        self.list_selection += 1;
                        if let Some(session_idx) = self.selected_session_index() {
                            self.session_manager.set_active(session_idx);
                        }
                    }
                }
            }
            InputResult::Select => {
                if !self.dialog.is_open() {
                    if self.list_selection == 0 {
                        self.dialog.open();
                    } else {
                        if let Some(session_idx) = self.selected_session_index() {
                            self.session_manager.set_active(session_idx);
                        }
                        self.focus = Focus::Terminal;
                    }
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
            InputResult::StartRename => {
                if let Some(session_idx) = self.selected_session_index() {
                    if let Some(session) = self.session_manager.sessions().get(session_idx) {
                        self.renaming = true;
                        self.rename_buffer = session.name.clone();
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_rename_input(&mut self, key: event::KeyEvent) -> Result<()> {
        match key.code {
            event::KeyCode::Esc => {
                self.renaming = false;
                self.rename_buffer.clear();
            }
            event::KeyCode::Enter => {
                if let Some(session_idx) = self.selected_session_index() {
                    if let Some(session) = self.session_manager.sessions_mut().get_mut(session_idx) {
                        let name = self.rename_buffer.trim();
                        if !name.is_empty() {
                            session.name = name.to_string();
                        }
                    }
                }
                self.renaming = false;
                self.rename_buffer.clear();
            }
            event::KeyCode::Backspace => {
                self.rename_buffer.pop();
            }
            event::KeyCode::Char(c) => {
                self.rename_buffer.push(c);
            }
            _ => {}
        }
        Ok(())
    }

    fn refresh_visible_sessions(&mut self) {
        let query = self.search_query.trim().to_lowercase();
        self.visible_sessions = self
            .session_manager
            .sessions()
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                if query.is_empty() {
                    true
                } else {
                    s.name.to_lowercase().contains(&query)
                        || s.agent_type.as_str().to_lowercase().contains(&query)
                }
            })
            .map(|(i, _)| i)
            .collect();
    }

    fn clamp_list_selection(&mut self) {
        let max = self.visible_sessions.len();
        if self.list_selection > max {
            self.list_selection = max;
        }
    }

    fn selected_session_index(&self) -> Option<usize> {
        if self.list_selection == 0 {
            None
        } else {
            self.visible_sessions.get(self.list_selection - 1).copied()
        }
    }

    fn index_in_visible(&self, session_index: usize) -> Option<usize> {
        self.visible_sessions.iter().position(|i| *i == session_index)
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
