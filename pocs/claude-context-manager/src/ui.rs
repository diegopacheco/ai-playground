use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs, Wrap};
use crate::app::{App, Dialog};
use crate::catalog::CatalogStatus;
use crate::model::{ArtifactKind, Health, Tab};

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(f.area());

    draw_header(f, chunks[0]);
    draw_tabs(f, app, chunks[1]);
    draw_content(f, app, chunks[2]);
    draw_footer(f, app, chunks[3]);

    if let Some(ref content) = app.preview_content {
        draw_preview_overlay(f, content);
    }

    if app.show_help {
        draw_help_overlay(f);
    }

    if let Some(ref dialog) = app.dialog {
        draw_dialog(f, app, dialog);
    }
}

fn draw_header(f: &mut Frame, area: Rect) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled("  Claude Context Manager ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("v0.1.0", Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
    f.render_widget(header, area);
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let all_tabs = Tab::all();
    let titles: Vec<Line> = all_tabs.iter()
        .map(|t| {
            let style = if *t == app.tab {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            Line::from(Span::styled(t.label(), style))
        })
        .collect();
    let tabs = Tabs::new(titles)
        .select(app.tab.index())
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED))
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(tabs, area);
}

fn draw_content(f: &mut Frame, app: &App, area: Rect) {
    match app.tab {
        Tab::Context => draw_context_dashboard(f, app, area),
        Tab::Catalog => draw_catalog(f, app, area),
        Tab::Backup => draw_backup(f, app, area),
        _ => draw_artifact_list(f, app, area),
    }
}

fn draw_context_dashboard(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(16),
            Constraint::Min(5),
        ])
        .split(area);

    let mcps = app.artifacts.iter().filter(|a| a.kind == ArtifactKind::Mcp).count();
    let hooks = app.artifacts.iter().filter(|a| a.kind == ArtifactKind::Hook).count();
    let commands = app.artifacts.iter().filter(|a| a.kind == ArtifactKind::Command).count();
    let agents = app.artifacts.iter().filter(|a| a.kind == ArtifactKind::Agent).count();
    let skills = app.artifacts.iter().filter(|a| a.kind == ArtifactKind::Skill).count();
    let context_files = app.artifacts.iter().filter(|a| a.kind == ArtifactKind::ContextFile).count();
    let memory_files = app.artifacts.iter().filter(|a| a.kind == ArtifactKind::MemoryFile).count();
    let total = app.artifacts.len();
    let healthy = app.artifacts.iter().filter(|a| matches!(a.health, Health::Active)).count();
    let warnings = app.artifacts.iter().filter(|a| matches!(a.health, Health::Warning(_))).count();
    let broken = app.artifacts.iter().filter(|a| matches!(a.health, Health::Broken(_))).count();
    let backups = app.backups.len();

    let bar_width = 30usize;
    let max_count = *[mcps, hooks, commands, agents, skills, context_files, memory_files].iter().max().unwrap_or(&1).max(&1);

    let make_bar = |count: usize, color: Color| -> Vec<Span> {
        let filled = if max_count > 0 { (count * bar_width) / max_count } else { 0 };
        let filled = filled.max(if count > 0 { 1 } else { 0 });
        vec![
            Span::styled("█".repeat(filled), Style::default().fg(color)),
            Span::styled("░".repeat(bar_width - filled), Style::default().fg(Color::DarkGray)),
            Span::styled(format!(" {}", count), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]
    };

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("   Claude Context Manager ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  {} total artifacts", total), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
    ];

    let categories: Vec<(&str, usize, Color)> = vec![
        ("  MCPs           ", mcps, Color::Magenta),
        ("  Hooks          ", hooks, Color::Blue),
        ("  Commands       ", commands, Color::Green),
        ("  Agents         ", agents, Color::Yellow),
        ("  Skills         ", skills, Color::Cyan),
        ("  Context Files  ", context_files, Color::White),
        ("  Memory Files   ", memory_files, Color::DarkGray),
    ];

    for (label, count, color) in categories {
        let mut spans = vec![Span::styled(label, Style::default().fg(color))];
        spans.extend(make_bar(count, color));
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  Health  ", Style::default().fg(Color::White)),
        Span::styled(format!("● {} active  ", healthy), Style::default().fg(Color::Green)),
        Span::styled(format!("● {} warning  ", warnings), Style::default().fg(Color::Yellow)),
        Span::styled(format!("○ {} broken  ", broken), Style::default().fg(Color::Red)),
        Span::styled(format!("  {} backups", backups), Style::default().fg(Color::DarkGray)),
    ]));

    let summary = Paragraph::new(lines)
        .block(Block::default()
            .title(" Dashboard ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)));
    f.render_widget(summary, chunks[0]);

    let items = app.current_items();
    let list_items: Vec<ListItem> = items.iter().enumerate().map(|(i, artifact)| {
        let (icon, icon_color) = match &artifact.health {
            Health::Active => ("●", Color::Green),
            Health::Warning(_) => ("●", Color::Yellow),
            Health::Broken(_) => ("○", Color::Red),
        };
        let selected = i == app.selection;
        let name_style = if selected {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        let line = Line::from(vec![
            Span::styled(if selected { " > " } else { "   " }, Style::default().fg(Color::Cyan)),
            Span::styled(icon, Style::default().fg(icon_color)),
            Span::raw(" "),
            Span::styled(&artifact.name, name_style),
            Span::styled(format!("  [{}]", artifact.scope.label()), Style::default().fg(Color::DarkGray)),
        ]);
        ListItem::new(line)
    }).collect();

    let list = List::new(list_items)
        .block(Block::default()
            .title(format!(" Context & Memory ({}) ", items.len()))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(list, chunks[1]);
}

fn draw_artifact_list(f: &mut Frame, app: &App, area: Rect) {
    let items = app.current_items();
    let list_items: Vec<ListItem> = items.iter().enumerate().map(|(i, artifact)| {
        let (icon, icon_color) = match &artifact.health {
            Health::Active => ("●", Color::Green),
            Health::Warning(_) => ("●", Color::Yellow),
            Health::Broken(_) => ("○", Color::Red),
        };

        let selected = i == app.selection;
        let name_style = if selected {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let line = Line::from(vec![
            Span::styled(if selected { " > " } else { "   " }, Style::default().fg(Color::Cyan)),
            Span::styled(icon, Style::default().fg(icon_color)),
            Span::raw(" "),
            Span::styled(&artifact.name, name_style),
            Span::styled(
                format!("  [{}]", artifact.scope.label()),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("  {}", artifact.kind.label()),
                Style::default().fg(Color::DarkGray),
            ),
            match &artifact.health {
                Health::Warning(msg) => Span::styled(format!("  {}", msg), Style::default().fg(Color::Yellow)),
                Health::Broken(msg) => Span::styled(format!("  {}", msg), Style::default().fg(Color::Red)),
                _ => Span::raw(""),
            },
        ]);
        ListItem::new(line)
    }).collect();

    let title = format!(" {} ({}) ", app.tab.label(), items.len());
    let search_title = if !app.search_query.is_empty() {
        format!(" {} ({}) [filter: {}] ", app.tab.label(), items.len(), app.search_query)
    } else {
        title
    };

    let list = List::new(list_items)
        .block(Block::default()
            .title(search_title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(list, area);
}

fn draw_catalog(f: &mut Frame, app: &App, area: Rect) {
    match &app.catalog.status {
        CatalogStatus::NotLoaded => {
            let msg = Paragraph::new(Line::from(vec![
                Span::styled("  Press ", Style::default().fg(Color::White)),
                Span::styled("[l]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(" to load catalog from GitHub or ", Style::default().fg(Color::White)),
                Span::styled("[i]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(" to load and install", Style::default().fg(Color::White)),
            ]))
            .block(Block::default()
                .title(" Catalog ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)));
            f.render_widget(msg, area);
        }
        CatalogStatus::Loading => {
            let msg = Paragraph::new(Span::styled("  Cloning repository...", Style::default().fg(Color::Yellow)))
                .block(Block::default()
                    .title(" Catalog ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)));
            f.render_widget(msg, area);
        }
        CatalogStatus::Error(e) => {
            let msg = Paragraph::new(Span::styled(format!("  Error: {}", e), Style::default().fg(Color::Red)))
                .block(Block::default()
                    .title(" Catalog ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)));
            f.render_widget(msg, area);
        }
        CatalogStatus::Loaded => {
            let list_items: Vec<ListItem> = app.catalog.items.iter().enumerate().map(|(i, item)| {
                let selected = i == app.selection;
                let installed_icon = if item.installed { "✓" } else { " " };
                let icon_color = if item.installed { Color::Green } else { Color::DarkGray };
                let name_style = if selected {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                let line = Line::from(vec![
                    Span::styled(if selected { " > " } else { "   " }, Style::default().fg(Color::Cyan)),
                    Span::styled(installed_icon, Style::default().fg(icon_color)),
                    Span::raw(" "),
                    Span::styled(&item.name, name_style),
                    Span::styled(format!("  [{}]", item.kind.label()), Style::default().fg(Color::DarkGray)),
                    if !item.description.is_empty() {
                        Span::styled(format!("  {}", item.description), Style::default().fg(Color::DarkGray))
                    } else {
                        Span::raw("")
                    },
                ]);
                ListItem::new(line)
            }).collect();

            let list = List::new(list_items)
                .block(Block::default()
                    .title(format!(" Catalog ({}) - [i] install ", app.catalog.items.len()))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)));
            f.render_widget(list, area);
        }
    }
}

fn draw_backup(f: &mut Frame, app: &App, area: Rect) {
    let list_items: Vec<ListItem> = app.backups.iter().enumerate().map(|(i, backup)| {
        let selected = i == app.selection;
        let name_style = if selected {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        let size = if backup.size_bytes > 1024 * 1024 {
            format!("{:.1} MB", backup.size_bytes as f64 / (1024.0 * 1024.0))
        } else if backup.size_bytes > 1024 {
            format!("{:.1} KB", backup.size_bytes as f64 / 1024.0)
        } else {
            format!("{} B", backup.size_bytes)
        };
        let line = Line::from(vec![
            Span::styled(if selected { " > " } else { "   " }, Style::default().fg(Color::Cyan)),
            Span::styled("●", Style::default().fg(Color::Green)),
            Span::raw(" "),
            Span::styled(&backup.created_at, name_style),
            Span::styled(format!("  ({})", size), Style::default().fg(Color::DarkGray)),
        ]);
        ListItem::new(line)
    }).collect();

    let list = List::new(list_items)
        .block(Block::default()
            .title(format!(" Backups ({}) - [b] create  [r] restore  [s] selective ", app.backups.len()))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(list, area);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let status = if app.searching {
        format!(" Search: {}_ ", app.search_query)
    } else if !app.status_msg.is_empty() {
        format!(" {} ", app.status_msg)
    } else {
        " [Tab] switch  [j/k] navigate  [d] delete  [b] backup  [/] search  [?] help  [q] quit ".to_string()
    };

    let status_color = if app.status_msg.starts_with("Error") {
        Color::Red
    } else if app.searching {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let footer = Paragraph::new(Span::styled(&status, Style::default().fg(status_color)))
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(footer, area);
}

fn draw_help_overlay(f: &mut Frame) {
    let area = centered_rect(60, 70, f.area());
    f.render_widget(Clear, area);

    let help_text = vec![
        Line::from(Span::styled("Key Bindings", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(vec![Span::styled("  Tab       ", Style::default().fg(Color::Yellow)), Span::raw("Next tab")]),
        Line::from(vec![Span::styled("  Shift+Tab ", Style::default().fg(Color::Yellow)), Span::raw("Previous tab")]),
        Line::from(vec![Span::styled("  j/k ↑↓    ", Style::default().fg(Color::Yellow)), Span::raw("Navigate up/down")]),
        Line::from(vec![Span::styled("  d         ", Style::default().fg(Color::Yellow)), Span::raw("Delete selected item")]),
        Line::from(vec![Span::styled("  b         ", Style::default().fg(Color::Yellow)), Span::raw("Create backup")]),
        Line::from(vec![Span::styled("  r         ", Style::default().fg(Color::Yellow)), Span::raw("Full restore (Backup tab)")]),
        Line::from(vec![Span::styled("  s         ", Style::default().fg(Color::Yellow)), Span::raw("Selective restore (Backup tab)")]),
        Line::from(vec![Span::styled("  l         ", Style::default().fg(Color::Yellow)), Span::raw("Load catalog (Catalog tab)")]),
        Line::from(vec![Span::styled("  i         ", Style::default().fg(Color::Yellow)), Span::raw("Install from catalog")]),
        Line::from(vec![Span::styled("  /         ", Style::default().fg(Color::Yellow)), Span::raw("Search/filter")]),
        Line::from(vec![Span::styled("  ?         ", Style::default().fg(Color::Yellow)), Span::raw("Toggle help")]),
        Line::from(vec![Span::styled("  q         ", Style::default().fg(Color::Yellow)), Span::raw("Quit")]),
        Line::from(""),
        Line::from(Span::styled("  Press any key to close", Style::default().fg(Color::DarkGray))),
    ];

    let help = Paragraph::new(help_text)
        .block(Block::default()
            .title(" Help ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)))
        .wrap(Wrap { trim: false });
    f.render_widget(help, area);
}

fn draw_dialog(f: &mut Frame, app: &App, dialog: &Dialog) {
    let area = centered_rect(50, 30, f.area());
    f.render_widget(Clear, area);

    match dialog {
        Dialog::ConfirmDelete(idx) => {
            let items = app.current_items();
            let name = items.get(*idx).map(|a| a.name.as_str()).unwrap_or("?");
            let text = vec![
                Line::from(""),
                Line::from(Span::styled(format!("  Remove '{}'?", name), Style::default().fg(Color::Yellow))),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  [y]", Style::default().fg(Color::Green)),
                    Span::raw(" Yes  "),
                    Span::styled("[n]", Style::default().fg(Color::Red)),
                    Span::raw(" No"),
                ]),
            ];
            let p = Paragraph::new(text)
                .block(Block::default()
                    .title(" Confirm Delete ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Red)));
            f.render_widget(p, area);
        }
        Dialog::ConfirmBackup => {
            let text = vec![
                Line::from(""),
                Line::from(Span::styled("  Create backup of all Claude config?", Style::default().fg(Color::Yellow))),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  [y]", Style::default().fg(Color::Green)),
                    Span::raw(" Yes  "),
                    Span::styled("[n]", Style::default().fg(Color::Red)),
                    Span::raw(" No"),
                ]),
            ];
            let p = Paragraph::new(text)
                .block(Block::default()
                    .title(" Confirm Backup ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan)));
            f.render_widget(p, area);
        }
        Dialog::ConfirmFullRestore(idx) => {
            let name = app.backups.get(*idx)
                .map(|b| b.created_at.as_str())
                .unwrap_or("?");
            let text = vec![
                Line::from(""),
                Line::from(Span::styled(format!("  Full restore from '{}'?", name), Style::default().fg(Color::Yellow))),
                Line::from(Span::styled("  Current config will be backed up first.", Style::default().fg(Color::DarkGray))),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  [y]", Style::default().fg(Color::Green)),
                    Span::raw(" Yes  "),
                    Span::styled("[n]", Style::default().fg(Color::Red)),
                    Span::raw(" No"),
                ]),
            ];
            let p = Paragraph::new(text)
                .block(Block::default()
                    .title(" Confirm Restore ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)));
            f.render_widget(p, area);
        }
        Dialog::SelectiveRestore(_) => {
            let items: Vec<Line> = app.restore_entries.iter().enumerate().map(|(i, entry)| {
                let checked = if app.restore_selected.get(i).copied().unwrap_or(false) { "[x]" } else { "[ ]" };
                let selected = i == app.selection;
                let style = if selected {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                Line::from(Span::styled(format!("  {} {}", checked, entry), style))
            }).collect();

            let mut lines = vec![
                Line::from(Span::styled("  Space=toggle  Enter=restore  Esc=cancel", Style::default().fg(Color::DarkGray))),
                Line::from(""),
            ];
            lines.extend(items);

            let p = Paragraph::new(lines)
                .block(Block::default()
                    .title(" Selective Restore ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)));

            let big_area = centered_rect(70, 80, f.area());
            f.render_widget(Clear, big_area);
            f.render_widget(p, big_area);
        }
        Dialog::InstallScope(_) => {
            let text = vec![
                Line::from(""),
                Line::from(Span::styled("  Install scope:", Style::default().fg(Color::Yellow))),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  [g]", Style::default().fg(Color::Green)),
                    Span::raw(" Global (~/.claude/)  "),
                    Span::styled("[p]", Style::default().fg(Color::Cyan)),
                    Span::raw(" Project (.claude/)"),
                ]),
                Line::from(""),
                Line::from(Span::styled("  Esc to cancel", Style::default().fg(Color::DarkGray))),
            ];
            let p = Paragraph::new(text)
                .block(Block::default()
                    .title(" Install ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Green)));
            f.render_widget(p, area);
        }
    }
}

fn draw_preview_overlay(f: &mut Frame, content: &str) {
    let area = centered_rect(80, 85, f.area());
    f.render_widget(Clear, area);
    let lines: Vec<Line> = content.lines()
        .take(100)
        .map(|l| Line::from(Span::styled(l, Style::default().fg(Color::White))))
        .collect();
    let preview = Paragraph::new(lines)
        .block(Block::default()
            .title(" Preview (Space to close) ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)))
        .wrap(Wrap { trim: false });
    f.render_widget(preview, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
