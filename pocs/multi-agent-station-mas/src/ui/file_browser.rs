use std::fs;
use std::path::PathBuf;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{List, ListItem, Paragraph},
};

pub struct FileBrowser {
    current_dir: PathBuf,
    entries: Vec<PathBuf>,
    selected: usize,
}

impl FileBrowser {
    pub fn new() -> Self {
        let current_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
        let mut browser = Self {
            current_dir: current_dir.clone(),
            entries: Vec::new(),
            selected: 0,
        };
        browser.refresh();
        browser
    }

    pub fn current_path(&self) -> PathBuf {
        self.current_dir.clone()
    }

    pub fn refresh(&mut self) {
        self.entries.clear();
        if let Ok(read_dir) = fs::read_dir(&self.current_dir) {
            let mut dirs: Vec<PathBuf> = read_dir
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .filter(|p| !p.file_name().map(|n| n.to_string_lossy().starts_with('.')).unwrap_or(false))
                .collect();
            dirs.sort();
            self.entries = dirs;
        }
        self.selected = 0;
    }

    pub fn up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn down(&mut self) {
        if self.selected + 1 < self.entries.len() {
            self.selected += 1;
        }
    }

    pub fn enter(&mut self) {
        if let Some(path) = self.entries.get(self.selected) {
            self.current_dir = path.clone();
            self.refresh();
        }
    }

    pub fn go_up(&mut self) {
        if let Some(parent) = self.current_dir.parent() {
            self.current_dir = parent.to_path_buf();
            self.refresh();
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Min(0),
                Constraint::Length(2),
            ])
            .split(area);

        let title = Paragraph::new("Select Working Directory:")
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(title, chunks[0]);

        let path_display = Paragraph::new(format!(" {}", self.current_dir.display()))
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(path_display, chunks[1]);

        let items: Vec<ListItem> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let name = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "?".to_string());
                let style = if i == self.selected {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                let prefix = if i == self.selected { "▶ " } else { "  " };
                ListItem::new(format!("{}📁 {}", prefix, name)).style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, chunks[2]);

        let help = Paragraph::new("↑/↓: Navigate | Enter: Open folder | Space: Select this folder | Backspace: Up | Esc: Back")
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(help, chunks[3]);
    }
}
