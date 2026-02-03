use std::path::PathBuf;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
};
use crate::agent::AgentType;
use super::FileBrowser;

#[derive(Debug, Clone, PartialEq)]
pub enum DialogState {
    Closed,
    SelectingAgent,
    SelectingDirectory,
}

pub struct NewSessionDialog {
    pub state: DialogState,
    pub selected_agent: usize,
    pub file_browser: FileBrowser,
}

impl NewSessionDialog {
    pub fn new() -> Self {
        Self {
            state: DialogState::Closed,
            selected_agent: 0,
            file_browser: FileBrowser::new(),
        }
    }

    pub fn open(&mut self) {
        self.state = DialogState::SelectingAgent;
        self.selected_agent = 0;
        self.file_browser = FileBrowser::new();
    }

    pub fn close(&mut self) {
        self.state = DialogState::Closed;
    }

    pub fn is_open(&self) -> bool {
        self.state != DialogState::Closed
    }

    pub fn selected_agent_type(&self) -> AgentType {
        AgentType::all()[self.selected_agent]
    }

    pub fn selected_directory(&self) -> PathBuf {
        self.file_browser.current_path()
    }

    pub fn next_agent(&mut self) {
        let agents = AgentType::all();
        self.selected_agent = (self.selected_agent + 1) % agents.len();
    }

    pub fn prev_agent(&mut self) {
        let agents = AgentType::all();
        self.selected_agent = if self.selected_agent == 0 {
            agents.len() - 1
        } else {
            self.selected_agent - 1
        };
    }

    pub fn confirm_agent(&mut self) {
        self.state = DialogState::SelectingDirectory;
    }

    pub fn back_to_agent(&mut self) {
        self.state = DialogState::SelectingAgent;
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        if !self.is_open() {
            return;
        }

        let dialog_width = 50.min(area.width.saturating_sub(4));
        let dialog_height = 20.min(area.height.saturating_sub(4));
        let x = (area.width.saturating_sub(dialog_width)) / 2;
        let y = (area.height.saturating_sub(dialog_height)) / 2;
        let dialog_area = Rect::new(x, y, dialog_width, dialog_height);

        frame.render_widget(Clear, dialog_area);

        let block = Block::default()
            .title(" New Session ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        frame.render_widget(block, dialog_area);

        let inner = Rect::new(
            dialog_area.x + 2,
            dialog_area.y + 2,
            dialog_area.width.saturating_sub(4),
            dialog_area.height.saturating_sub(4),
        );

        match self.state {
            DialogState::SelectingAgent => {
                self.render_agent_selection(frame, inner);
            }
            DialogState::SelectingDirectory => {
                self.file_browser.render(frame, inner);
            }
            DialogState::Closed => {}
        }
    }

    fn render_agent_selection(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(0), Constraint::Length(2)])
            .split(area);

        let title = Paragraph::new("Select Agent Type:")
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(title, chunks[0]);

        let agents = AgentType::all();
        let items: Vec<ListItem> = agents
            .iter()
            .enumerate()
            .map(|(i, agent)| {
                let style = if i == self.selected_agent {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                let prefix = if i == self.selected_agent { "▶ " } else { "  " };
                ListItem::new(format!("{}{}", prefix, agent))
                    .style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, chunks[1]);

        let help = Paragraph::new("↑/↓: Select | Enter: Confirm | Esc: Cancel")
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(help, chunks[2]);
    }
}
