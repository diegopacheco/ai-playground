use crate::banner;
use crate::catalog::Item;
use crate::detail::{self, Detail};
use crate::text::{truncate, wrap};
use crate::error::Result;
use crate::kind::{Kind, Target};
use crate::picker::Picker;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind};
use crossterm::style::{Color, Stylize};
use crossterm::terminal::{self, ClearType};
use crossterm::{cursor, execute, queue};
use std::io::{Stdout, Write, stdout};
use std::path::Path;

struct Screen {
    out: Stdout,
}

impl Screen {
    fn open() -> Result<Self> {
        terminal::enable_raw_mode()?;
        let mut out = stdout();
        execute!(out, terminal::EnterAlternateScreen, cursor::Hide, event::EnableMouseCapture)?;
        Ok(Screen { out })
    }

    fn draw(&mut self, lines: &[String]) -> Result<()> {
        queue!(self.out, cursor::MoveTo(0, 0), terminal::Clear(ClearType::All))?;
        write!(self.out, "{}", lines.join("\r\n"))?;
        self.out.flush()?;
        Ok(())
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        let _ = execute!(self.out, event::DisableMouseCapture, cursor::Show, terminal::LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

enum Input {
    Key(KeyEvent),
    Click(u16, u16),
    ScrollUp,
    ScrollDown,
    Resize,
}

fn next_input() -> Result<Input> {
    loop {
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => return Ok(Input::Key(key)),
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::Down(MouseButton::Left) => return Ok(Input::Click(mouse.column, mouse.row)),
                MouseEventKind::ScrollUp => return Ok(Input::ScrollUp),
                MouseEventKind::ScrollDown => return Ok(Input::ScrollDown),
                _ => {}
            },
            Event::Resize(_, _) => return Ok(Input::Resize),
            _ => {}
        }
    }
}

fn is_quit(key: &KeyEvent) -> bool {
    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)
}

fn header(lines: &mut Vec<String>, height: u16) {
    if height >= 22 {
        lines.push(String::new());
        lines.extend(banner::render(0.15).split("\r\n").filter(|l| !l.is_empty()).map(str::to_string));
    } else {
        lines.push(format!("   {}", "RAM".with(banner::hue(0.0)).bold()));
    }
    lines.push(String::new());
}

const TARGET_OPTIONS: [(&str, &str); 3] = [
    ("Claude", "~/.claude  skills, commands, agents, plugins"),
    ("Codex", "~/.codex   skills, prompts"),
    ("Both", "Claude and Codex"),
];

pub fn choose_targets(source: &str, count: usize) -> Result<Option<Vec<Target>>> {
    let mut screen = Screen::open()?;
    let mut cursor = 0usize;
    let mut option_top: u16;
    loop {
        let (width, height) = terminal::size()?;
        let mut lines = Vec::new();
        header(&mut lines, height);
        lines.push(format!("   {} {} from {}", "✔".with(Color::Green), format!("{count} items").bold(), source.with(Color::DarkGrey)));
        lines.push(String::new());
        lines.push(format!("   {}", "Where do you want to install globally?".bold()));
        lines.push(String::new());
        option_top = lines.len() as u16;
        for (i, (name, detail)) in TARGET_OPTIONS.iter().enumerate() {
            let active = i == cursor;
            let marker = if active { "❯".with(banner::hue(0.55)).bold() } else { " ".stylize() };
            let label = if active { format!("{name:<8}").with(Color::White).bold() } else { format!("{name:<8}").with(Color::Grey) };
            lines.push(format!("   {marker} {label} {}", detail.with(Color::DarkGrey)));
        }
        lines.push(String::new());
        lines.push(help(width, "↑↓ move   enter select   click choose   esc quit"));
        screen.draw(&lines)?;
        match next_input()? {
            Input::Key(key) if is_quit(&key) || key.code == KeyCode::Esc || key.code == KeyCode::Char('q') => return Ok(None),
            Input::Key(key) => match key.code {
                KeyCode::Up | KeyCode::Char('k') => cursor = (cursor + 2) % 3,
                KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => cursor = (cursor + 1) % 3,
                KeyCode::Char('1') => cursor = 0,
                KeyCode::Char('2') => cursor = 1,
                KeyCode::Char('3') => cursor = 2,
                KeyCode::Enter => return Ok(Some(targets_at(cursor))),
                _ => {}
            },
            Input::Click(_, row) if row >= option_top && row < option_top + 3 => return Ok(Some(targets_at((row - option_top) as usize))),
            _ => {}
        }
    }
}

fn targets_at(index: usize) -> Vec<Target> {
    match index {
        0 => vec![Target::Claude],
        1 => vec![Target::Codex],
        _ => vec![Target::Claude, Target::Codex],
    }
}

fn kind_color(kind: Kind) -> Color {
    match kind {
        Kind::Skill => Color::Rgb { r: 110, g: 200, b: 255 },
        Kind::Command => Color::Rgb { r: 230, g: 130, b: 255 },
        Kind::Agent => Color::Rgb { r: 255, g: 210, b: 100 },
        Kind::Plugin => Color::Rgb { r: 120, g: 235, b: 150 },
    }
}

const CHECKBOX_COLUMNS: std::ops::Range<u16> = 4..10;

fn help(width: u16, text: &str) -> String {
    format!("   {}", truncate(text, (width as usize).saturating_sub(4)).with(Color::DarkGrey))
}

fn rule(width: u16) -> String {
    format!("   {}", "─".repeat(width.saturating_sub(6) as usize).with(Color::DarkGrey))
}

fn kind_chips(picker: &Picker, kinds: &[Kind], items: &[Item]) -> String {
    let mut chips = Vec::new();
    let all = format!(" all {} ", items.len());
    chips.push(if picker.tag().is_none() { all.on(banner::hue(0.55)).with(Color::Black).bold().to_string() } else { all.with(Color::Grey).to_string() });
    for kind in kinds {
        let count = items.iter().filter(|i| i.kind == *kind).count();
        let chip = format!(" {}s {count} ", kind.label());
        let active = picker.tag() == Some(kind.label());
        chips.push(if active { chip.on(kind_color(*kind)).with(Color::Black).bold().to_string() } else { chip.with(kind_color(*kind)).to_string() });
    }
    chips.join(" ")
}

fn next_kind(picker: &Picker, kinds: &[Kind], forward: bool) -> Option<String> {
    let mut cycle: Vec<Option<String>> = vec![None];
    cycle.extend(kinds.iter().map(|k| Some(k.label().to_string())));
    let current = cycle.iter().position(|t| t.as_deref() == picker.tag()).unwrap_or(0);
    let next = if forward { (current + 1) % cycle.len() } else { (current + cycle.len() - 1) % cycle.len() };
    cycle[next].clone()
}

pub fn choose_items(items: &[Item], targets: &[Target], home: &Path) -> Result<Option<Vec<usize>>> {
    let mut screen = Screen::open()?;
    let mut picker = Picker::new(items.iter().map(Item::entry).collect());
    let kinds: Vec<Kind> = Kind::ALL.into_iter().filter(|k| items.iter().any(|i| i.kind == *k)).collect();
    let mut offset = 0usize;
    let name_width = items.iter().map(|i| i.name.chars().count()).max().unwrap_or(10).min(30);
    let target_text = targets.iter().map(|t| t.label()).collect::<Vec<_>>().join(" + ");
    loop {
        let (width, height) = terminal::size()?;
        let mut lines = Vec::new();
        header(&mut lines, height);
        lines.push(format!(
            "   {} {}   {} {}   {} {}",
            "target".with(Color::DarkGrey),
            target_text.as_str().with(Color::White).bold(),
            "showing".with(Color::DarkGrey),
            format!("{}/{}", picker.visible().len(), items.len()).with(Color::White),
            "selected".with(Color::DarkGrey),
            picker.chosen().len().to_string().with(Color::Green).bold()
        ));
        lines.push(format!("   {}  {}", "filter".with(Color::DarkGrey), kind_chips(&picker, &kinds, items)));
        let query = if picker.query().is_empty() {
            "type to filter by name, kind, path or description".with(Color::DarkGrey).italic().to_string()
        } else {
            picker.query().with(Color::White).bold().to_string()
        };
        lines.push(format!("   {} {}{}", "⌕ search".with(banner::hue(0.5)).bold(), "▏".with(Color::Grey), query));
        lines.push(rule(width));
        let list_top = lines.len();
        let rows = (height as usize).saturating_sub(list_top + 3).max(3);
        if picker.cursor() < offset {
            offset = picker.cursor();
        } else if picker.cursor() >= offset + rows {
            offset = picker.cursor() + 1 - rows;
        }
        offset = offset.min(picker.visible().len().saturating_sub(rows.min(picker.visible().len())));
        if picker.visible().is_empty() {
            lines.push(format!("   {}", "no matches".with(Color::DarkGrey).italic()));
        }
        for (row, &index) in picker.visible().iter().enumerate().skip(offset).take(rows) {
            let item = &items[index];
            let active = row == picker.cursor();
            let marker = if active { "❯".with(banner::hue(0.55)).bold() } else { " ".stylize() };
            let checked = picker.is_selected(index);
            let box_text = if checked { "[✔]".with(Color::Green).bold() } else { "[ ]".with(Color::DarkGrey) };
            let kind = format!("{:<7}", item.kind.label()).with(kind_color(item.kind));
            let name_text = format!("{:<name_width$}", truncate(&item.name, name_width));
            let name = if active || checked { name_text.with(Color::White).bold() } else { name_text.with(Color::Grey) };
            let used = 3 + 2 + 4 + 8 + name_width + 2 + 1;
            let description = truncate(&item.description, (width as usize).saturating_sub(used + 1));
            lines.push(format!("   {marker} {box_text} {kind} {name}  {}", description.with(Color::DarkGrey)));
        }
        while lines.len() < list_top + rows {
            lines.push(String::new());
        }
        lines.push(rule(width));
        lines.push(help(width, "tab kind  ↑↓ move  space/click box toggle  →/click name details  ctrl+a all  enter install  esc quit"));
        screen.draw(&lines)?;
        let open_detail = match next_input()? {
            Input::Key(key) if is_quit(&key) => return Ok(None),
            Input::Key(key) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                match key.code {
                    KeyCode::Char('a') => picker.toggle_all_visible(),
                    KeyCode::Char('u') => picker.clear_query(),
                    KeyCode::Char('j') | KeyCode::Char('n') => picker.down(),
                    KeyCode::Char('k') | KeyCode::Char('p') => picker.up(),
                    _ => {}
                }
                key.code == KeyCode::Char('o')
            }
            Input::Key(key) => match key.code {
                KeyCode::Esc if picker.query().is_empty() && picker.tag().is_none() => return Ok(None),
                KeyCode::Esc => {
                    picker.clear_query();
                    picker.set_tag(None);
                    false
                }
                KeyCode::Up => {
                    picker.up();
                    false
                }
                KeyCode::Down => {
                    picker.down();
                    false
                }
                KeyCode::Tab => {
                    picker.set_tag(next_kind(&picker, &kinds, true));
                    false
                }
                KeyCode::BackTab => {
                    picker.set_tag(next_kind(&picker, &kinds, false));
                    false
                }
                KeyCode::Right => true,
                KeyCode::Char(' ') => {
                    picker.toggle();
                    false
                }
                KeyCode::Backspace => {
                    picker.backspace();
                    false
                }
                KeyCode::Enter => {
                    if picker.chosen().is_empty() {
                        picker.toggle();
                    }
                    if !picker.chosen().is_empty() {
                        return Ok(Some(picker.chosen()));
                    }
                    false
                }
                KeyCode::Char(c) => {
                    picker.type_char(c);
                    false
                }
                _ => false,
            },
            Input::Click(column, row) => {
                let row = row as usize;
                let target = offset + row.wrapping_sub(list_top);
                if row >= list_top && row < list_top + rows && target < picker.visible().len() {
                    picker.set_cursor(target);
                    if CHECKBOX_COLUMNS.contains(&column) {
                        picker.toggle();
                        false
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
            Input::ScrollUp => {
                picker.up();
                false
            }
            Input::ScrollDown => {
                picker.down();
                false
            }
            Input::Resize => false,
        };
        if open_detail && let Some(&index) = picker.visible().get(picker.cursor()) {
            let detail = detail::load(&items[index], targets, home)?;
            match show_detail(&mut screen, &items[index], &detail, picker.is_selected(index))? {
                DetailExit::Back { toggled } => {
                    if toggled {
                        picker.toggle();
                    }
                }
                DetailExit::Install { toggled } => {
                    if toggled {
                        picker.toggle();
                    }
                    if picker.chosen().is_empty() {
                        picker.toggle();
                    }
                    return Ok(Some(picker.chosen()));
                }
                DetailExit::Quit => return Ok(None),
            }
        }
    }
}

enum DetailExit {
    Back { toggled: bool },
    Install { toggled: bool },
    Quit,
}

fn style_body(detail: &Detail, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_code = false;
    let mut in_front = false;
    for (n, line) in detail.body.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            out.push(format!("   {}", truncate(line, width).with(Color::DarkGrey)));
            continue;
        }
        if trimmed == "---" && (n == 0 || in_front) {
            in_front = !in_front;
            out.push(format!("   {}", "┄┄┄".with(Color::DarkGrey)));
            continue;
        }
        if in_code {
            out.push(format!("   {}", truncate(line, width).with(Color::Rgb { r: 160, g: 220, b: 160 })));
            continue;
        }
        for piece in wrap(line, width) {
            let styled = if in_front {
                match piece.split_once(':') {
                    Some((key, value)) if !key.starts_with(' ') => format!("{}:{}", key.with(banner::hue(0.55)).bold(), value.with(Color::Grey)),
                    _ => piece.with(Color::Grey).to_string(),
                }
            } else if trimmed.starts_with('#') {
                piece.with(banner::hue(0.78)).bold().to_string()
            } else if trimmed.starts_with("* ") || trimmed.starts_with("- ") || trimmed.starts_with("| ") {
                piece.with(Color::White).to_string()
            } else {
                piece.with(Color::Grey).to_string()
            };
            out.push(format!("   {styled}"));
        }
    }
    out
}

fn show_detail(screen: &mut Screen, item: &Item, detail: &Detail, selected: bool) -> Result<DetailExit> {
    let mut scroll = 0usize;
    let mut toggled = false;
    loop {
        let (width, height) = terminal::size()?;
        let content_width = (width as usize).saturating_sub(8);
        let checked = selected != toggled;
        let mut lines = vec![String::new()];
        let box_text = if checked { "[✔]".with(Color::Green).bold() } else { "[ ]".with(Color::DarkGrey) };
        lines.push(format!(
            "   {box_text} {}  {}",
            format!(" {} ", item.kind.label()).on(kind_color(item.kind)).with(Color::Black).bold(),
            item.name.as_str().with(Color::White).bold()
        ));
        for piece in wrap(&item.description, content_width) {
            lines.push(format!("   {}", piece.with(Color::Grey).italic()));
        }
        lines.push(String::new());
        for (key, value) in &detail.facts {
            lines.push(format!("   {} {}", format!("{key:<10}").with(Color::DarkGrey), truncate(value, content_width.saturating_sub(11)).with(Color::White)));
        }
        if !detail.files.is_empty() {
            let files = detail.files.join("  ");
            for (n, piece) in wrap(&files, content_width.saturating_sub(11)).into_iter().enumerate() {
                let label = if n == 0 { format!("{:<10}", format!("files {}", detail.files.len())) } else { " ".repeat(10) };
                lines.push(format!("   {} {}", label.with(Color::DarkGrey), piece.with(Color::Grey)));
            }
        }
        lines.push(rule(width));
        let body = style_body(detail, content_width);
        let rows = (height as usize).saturating_sub(lines.len() + 2).max(3);
        let max_scroll = body.len().saturating_sub(rows);
        scroll = scroll.min(max_scroll);
        lines.extend(body.iter().skip(scroll).take(rows).cloned());
        while lines.len() < (height as usize).saturating_sub(2) {
            lines.push(String::new());
        }
        let position = if body.is_empty() { "empty".to_string() } else { format!("{}-{} of {}", scroll + 1, (scroll + rows).min(body.len()), body.len()) };
        lines.push(rule(width));
        lines.push(help(width, &format!("{position}   ↑↓ pgup pgdn scroll   space toggle   esc/← back   enter install")));
        screen.draw(&lines)?;
        match next_input()? {
            Input::Key(key) if is_quit(&key) => return Ok(DetailExit::Quit),
            Input::Key(key) => match key.code {
                KeyCode::Esc | KeyCode::Left | KeyCode::Backspace | KeyCode::Char('q') => return Ok(DetailExit::Back { toggled }),
                KeyCode::Enter => return Ok(DetailExit::Install { toggled }),
                KeyCode::Char(' ') => toggled = !toggled,
                KeyCode::Up | KeyCode::Char('k') => scroll = scroll.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => scroll = (scroll + 1).min(max_scroll),
                KeyCode::PageUp => scroll = scroll.saturating_sub(rows),
                KeyCode::PageDown => scroll = (scroll + rows).min(max_scroll),
                KeyCode::Home | KeyCode::Char('g') => scroll = 0,
                KeyCode::End | KeyCode::Char('G') => scroll = max_scroll,
                _ => {}
            },
            Input::ScrollUp => scroll = scroll.saturating_sub(3),
            Input::ScrollDown => scroll = (scroll + 3).min(max_scroll),
            Input::Click(column, row) if row == 1 && CHECKBOX_COLUMNS.contains(&column) => toggled = !toggled,
            _ => {}
        }
    }
}
