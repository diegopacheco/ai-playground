use crate::error::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Ident(String),
    Str(String),
    Equals,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Eof,
}

impl TokenKind {
    pub fn describe(&self) -> &'static str {
        match self {
            TokenKind::Ident(_) => "identifier",
            TokenKind::Str(_) => "string",
            TokenKind::Equals => "'='",
            TokenKind::LBrace => "'{'",
            TokenKind::RBrace => "'}'",
            TokenKind::LBracket => "'['",
            TokenKind::RBracket => "']'",
            TokenKind::Comma => "','",
            TokenKind::Eof => "end of file",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub col: usize,
}

struct Cursor {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Cursor {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, Error> {
    let mut cur = Cursor {
        chars: src.chars().collect(),
        pos: 0,
        line: 1,
        col: 1,
    };
    let mut tokens = Vec::new();
    while let Some(c) = cur.peek() {
        let (line, col) = (cur.line, cur.col);
        let kind = match c {
            ' ' | '\t' | '\r' | '\n' => {
                cur.bump();
                continue;
            }
            '#' => {
                while cur.peek().is_some_and(|c| c != '\n') {
                    cur.bump();
                }
                continue;
            }
            '=' | '{' | '}' | '[' | ']' | ',' => {
                cur.bump();
                punctuation(c)
            }
            '"' => TokenKind::Str(read_string(&mut cur)?),
            c if c.is_ascii_alphabetic() || c == '_' => TokenKind::Ident(read_ident(&mut cur)),
            other => return Err(Error::new(line, col, format!("unexpected character '{other}'"))),
        };
        tokens.push(Token { kind, line, col });
    }
    tokens.push(Token {
        kind: TokenKind::Eof,
        line: cur.line,
        col: cur.col,
    });
    Ok(tokens)
}

fn punctuation(c: char) -> TokenKind {
    match c {
        '=' => TokenKind::Equals,
        '{' => TokenKind::LBrace,
        '}' => TokenKind::RBrace,
        '[' => TokenKind::LBracket,
        ']' => TokenKind::RBracket,
        _ => TokenKind::Comma,
    }
}

fn read_ident(cur: &mut Cursor) -> String {
    let mut out = String::new();
    while let Some(c) = cur.peek().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-') {
        out.push(c);
        cur.bump();
    }
    out
}

fn read_string(cur: &mut Cursor) -> Result<String, Error> {
    let (line, col) = (cur.line, cur.col);
    cur.bump();
    let mut out = String::new();
    loop {
        match cur.peek() {
            None | Some('\n') => return Err(Error::new(line, col, "unterminated string")),
            Some('"') => {
                cur.bump();
                return Ok(out);
            }
            Some('\\') => {
                let (el, ec) = (cur.line, cur.col);
                cur.bump();
                match cur.peek() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    None | Some('\n') => return Err(Error::new(line, col, "unterminated string")),
                    Some(other) => return Err(Error::new(el, ec, format!("invalid escape '\\{other}'"))),
                }
                cur.bump();
            }
            Some(c) => {
                out.push(c);
                cur.bump();
            }
        }
    }
}
