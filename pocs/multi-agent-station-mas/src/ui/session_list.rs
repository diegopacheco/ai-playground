use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use crate::session::Session;

pub fn render_session_list(
    frame: &mut Frame,
    area: Rect,
    sessions: &[Session],
    visible_indices: &[usize],
    active_index: Option<usize>,
    list_selection: usize,
    list_focused: bool,
    search_query: &str,
    renaming: bool,
    rename_buffer: &str,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(0)])
        .split(area);

    let search_style = if list_focused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let search_text = if renaming {
        format!(" Rename: {}", rename_buffer)
    } else {
        format!(" Search: {}", search_query)
    };

    let search = Paragraph::new(search_text)
        .style(search_style);
    frame.render_widget(search, chunks[0]);

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

    for (i, session_index) in visible_indices.iter().enumerate() {
        let session = &sessions[*session_index];
        let is_selected = list_focused && list_selection == i + 1;
        let is_active = Some(*session_index) == active_index;
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
            Span::styled(format!("{}", session.name), style),
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
    frame.render_widget(list, chunks[1]);
}
