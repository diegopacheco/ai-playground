use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error,
    Warning,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Level::Error => write!(f, "error"),
            Level::Warning => write!(f, "warning"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub level: Level,
    pub code: &'static str,
    pub line: usize,
    pub col: usize,
    pub message: String,
}

impl Diagnostic {
    pub fn error(code: &'static str, line: usize, col: usize, message: String) -> Self {
        Self { level: Level::Error, code, line, col, message }
    }

    pub fn warning(code: &'static str, line: usize, col: usize, message: String) -> Self {
        Self { level: Level::Warning, code, line, col, message }
    }

    pub fn render(&self, path: &str) -> String {
        format!("{path}:{}:{}: {}[{}]: {}", self.line, self.col, self.level, self.code, self.message)
    }
}
