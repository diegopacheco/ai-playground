use crate::banner;
use crate::catalog::Item;
use crate::error::Result;
use crate::kind::{Kind, Target};
use crate::picker::Picker;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind};
use crossterm::style::{Color, Stylize};
use crossterm::terminal::{self, ClearType};
use crossterm::{cursor, execute, queue};
use std::io::{Stdout, Write, stdout};

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
    Click(u16),
    ScrollUp,
    ScrollDown,
    Resize,
}

fn next_input() -> Result<Input> {
    loop {
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => return Ok(Input::Key(key)),
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::Down(MouseButton::Left) => return Ok(Input::Click(mouse.row)),
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
        let (_, height) = terminal::size()?;
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
        lines.push(format!("   {}", "↑↓ move   enter select   click choose   esc quit".with(Color::DarkGrey)));
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
            Input::Click(row) if row >= option_top && row < option_top + 3 => return Ok(Some(targets_at((row - option_top) as usize))),
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

fn char_width(c: char) -> usize {
    match c as u32 {
        0x200D | 0xFE00..=0xFE0F | 0x0300..=0x036F => 0,
        0x1100..=0x115F | 0x2600..=0x27BF | 0x2E80..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF | 0xFF00..=0xFF60 | 0x1F000..=0x1FAFF => 2,
        _ => 1,
    }
}

pub fn truncate(text: &str, width: usize) -> String {
    let clean: String = text.chars().filter(|c| char_width(*c) > 0 || *c == ' ').collect();
    if clean.chars().map(char_width).sum::<usize>() <= width {
        return clean;
    }
    let mut used = 0;
    let mut cut = String::new();
    for c in clean.chars() {
        let w = char_width(c);
        if used + w + 1 > width {
            break;
        }
        used += w;
        cut.push(c);
    }
    cut.push('…');
    cut
}

pub fn choose_items(items: &[Item], targets: &[Target]) -> Result<Option<Vec<usize>>> {
    let mut screen = Screen::open()?;
    let mut picker = Picker::new(items.iter().map(Item::entry).collect());
    let mut offset = 0usize;
    let name_width = items.iter().map(|i| i.name.chars().count()).max().unwrap_or(10).min(30);
    let target_text = targets.iter().map(|t| t.label()).collect::<Vec<_>>().join(" + ");
    loop {
        let (width, height) = terminal::size()?;
        let mut lines = Vec::new();
        header(&mut lines, height);
        let chosen = picker.chosen().len();
        lines.push(format!(
            "   {} {}   {} {}   {} {}",
            "target".with(Color::DarkGrey),
            target_text.as_str().with(Color::White).bold(),
            "showing".with(Color::DarkGrey),
            format!("{}/{}", picker.visible().len(), items.len()).with(Color::White),
            "selected".with(Color::DarkGrey),
            chosen.to_string().with(Color::Green).bold()
        ));
        lines.push(format!("   {} {}{}", "⌕ search".with(banner::hue(0.5)).bold(), picker.query().with(Color::White).bold(), "▏".with(Color::Grey).slow_blink()));
        lines.push(format!("   {}", "─".repeat(width.saturating_sub(6) as usize).with(Color::DarkGrey)));
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
        lines.push(format!("   {}", "─".repeat(width.saturating_sub(6) as usize).with(Color::DarkGrey)));
        lines.push(format!(
            "   {}",
            "type to search   ↑↓ move   space/click toggle   ctrl+a all   enter install   esc clear/quit".with(Color::DarkGrey)
        ));
        screen.draw(&lines)?;
        match next_input()? {
            Input::Key(key) if is_quit(&key) => return Ok(None),
            Input::Key(key) if key.modifiers.contains(KeyModifiers::CONTROL) => match key.code {
                KeyCode::Char('a') => picker.toggle_all_visible(),
                KeyCode::Char('u') => picker.clear_query(),
                KeyCode::Char('j') | KeyCode::Char('n') => picker.down(),
                KeyCode::Char('k') | KeyCode::Char('p') => picker.up(),
                _ => {}
            },
            Input::Key(key) => match key.code {
                KeyCode::Esc if picker.query().is_empty() => return Ok(None),
                KeyCode::Esc => picker.clear_query(),
                KeyCode::Up => picker.up(),
                KeyCode::Down => picker.down(),
                KeyCode::Char(' ') | KeyCode::Tab => picker.toggle(),
                KeyCode::Backspace => picker.backspace(),
                KeyCode::Enter => {
                    if picker.chosen().is_empty() {
                        picker.toggle();
                    }
                    if !picker.chosen().is_empty() {
                        return Ok(Some(picker.chosen()));
                    }
                }
                KeyCode::Char(c) => picker.type_char(c),
                _ => {}
            },
            Input::Click(row) => {
                let row = row as usize;
                if row >= list_top && row < list_top + rows {
                    let target = offset + row - list_top;
                    if target < picker.visible().len() {
                        picker.set_cursor(target);
                        picker.toggle();
                    }
                }
            }
            Input::ScrollUp => picker.up(),
            Input::ScrollDown => picker.down(),
            Input::Resize => {}
        }
    }
}
