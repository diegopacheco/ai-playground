use crossterm::style::{Color, Stylize};
use std::fmt::Display;
use std::io::IsTerminal;
use std::sync::OnceLock;

static ENABLED: OnceLock<bool> = OnceLock::new();

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none())
}

pub fn paint(text: impl Display, color: Color) -> String {
    if enabled() { text.to_string().with(color).to_string() } else { text.to_string() }
}

pub fn strong(text: impl Display, color: Color) -> String {
    if enabled() { text.to_string().with(color).bold().to_string() } else { text.to_string() }
}

pub fn bold(text: impl Display) -> String {
    if enabled() { text.to_string().bold().to_string() } else { text.to_string() }
}
