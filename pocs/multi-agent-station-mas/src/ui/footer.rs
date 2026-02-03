use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

pub fn render_footer(frame: &mut Frame, area: Rect, session_count: usize, error: Option<&str>) {
    let mut spans = vec![
        Span::raw(" Sessions: "),
        Span::styled(format!("{}", session_count), Style::default().fg(Color::Cyan)),
        Span::raw(" | "),
        Span::styled("Cmd+T", Style::default().fg(Color::Yellow)),
        Span::raw(": New | "),
        Span::styled("Ctrl+1-9", Style::default().fg(Color::Yellow)),
        Span::raw(": Switch | "),
        Span::styled("Ctrl+W", Style::default().fg(Color::Yellow)),
        Span::raw(": Quit"),
    ];

    if let Some(message) = error {
        spans.push(Span::raw(" | "));
        spans.push(Span::styled(message.to_string(), Style::default().fg(Color::Red)));
    }

    let line = Line::from(spans);

    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(line).block(block);
    frame.render_widget(paragraph, area);
}
