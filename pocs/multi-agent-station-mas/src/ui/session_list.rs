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
    list_selection: usize,
    list_focused: bool,
) {
    let new_btn_style = if list_focused && list_selection == 0 {
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Green)
    };
    let new_prefix = if list_focused && list_selection == 0 { "▶ " } else { "  " };
    
    let mut items: Vec<ListItem> = vec![
        ListItem::new(Line::from(vec![
            Span::raw(new_prefix),
            Span::styled("[+] New Session", new_btn_style),
        ])),
    ];

    for (i, session) in sessions.iter().enumerate() {
        let is_selected = list_focused && list_selection == i + 1;
        let is_active = Some(i) == active_index;
        let prefix = if is_selected || is_active { "▶ " } else { "  " };
        let style = if is_active {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else if is_selected {
            Style::default().fg(Color::Yellow)
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
