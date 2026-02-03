use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph},
};
use crate::session::Session;

pub fn render_terminal(
    frame: &mut Frame,
    area: Rect,
    session: Option<&Session>,
    focused: bool,
) {
    let border_style = if focused {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let title = session
        .map(|s| format!(" {} [PID: {}] - {} ", s.name, s.pid, s.working_dir.display()))
        .unwrap_or_else(|| " No Session ".to_string());

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(border_style);

    match session {
        Some(s) => {
            let rows = s.screen_rows();
            let lines: Vec<Line> = rows.iter().map(|r| Line::raw(r.as_str())).collect();
            let paragraph = Paragraph::new(lines).block(block);
            frame.render_widget(paragraph, area);
        }
        None => {
            let placeholder = Paragraph::new("Press Cmd+T or click [+] to create a new session")
                .style(Style::default().fg(Color::DarkGray))
                .block(block);
            frame.render_widget(placeholder, area);
        }
    }
}
