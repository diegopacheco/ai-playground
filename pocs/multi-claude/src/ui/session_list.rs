use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem},
};
use crate::session::Session;

pub fn render_session_list(
    frame: &mut Frame,
    area: Rect,
    sessions: &[Session],
    active_index: Option<usize>,
    list_focused: bool,
) {
    let mut items: Vec<ListItem> = vec![
        ListItem::new(Line::from(vec![
            Span::styled("[+] ", Style::default().fg(Color::Green)),
            Span::raw("New Session"),
        ])),
    ];

    for (i, session) in sessions.iter().enumerate() {
        let prefix = if Some(i) == active_index { "▶ " } else { "  " };
        let style = if Some(i) == active_index {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else if session.exited {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };

        items.push(ListItem::new(Line::from(vec![
            Span::raw(prefix),
            Span::styled(format!("{}", session.agent_type), style),
            Span::styled(format!(" [{}]", session.pid), Style::default().fg(Color::DarkGray)),
        ])));
    }

    let border_style = if list_focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .title(" Sessions ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}
