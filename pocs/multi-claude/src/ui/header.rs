use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use chrono::Local;

pub fn render_header(frame: &mut Frame, area: Rect) {
    let now = Local::now();
    let time_str = now.format("%H:%M:%S").to_string();
    let title = "Multi-Claude";
    let padding = area.width.saturating_sub(title.len() as u16 + time_str.len() as u16 + 4) / 2;
    let left_pad = " ".repeat(padding as usize);
    let right_pad = " ".repeat((area.width.saturating_sub(padding + title.len() as u16 + time_str.len() as u16 + 4)) as usize);

    let line = Line::from(vec![
        Span::raw(left_pad),
        Span::styled(title, Style::default().fg(Color::Cyan)),
        Span::raw(right_pad),
        Span::styled(time_str, Style::default().fg(Color::Yellow)),
        Span::raw("  "),
    ]);

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(line).block(block);
    frame.render_widget(paragraph, area);
}
