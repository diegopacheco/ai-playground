use std::collections::BTreeMap;

use crate::error::Error;
use crate::lexer::{Token, TokenKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Str(String),
    List(Vec<String>),
    Map(BTreeMap<String, String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    pub name: String,
    pub line: usize,
    pub col: usize,
    pub value: Value,
    pub value_line: usize,
    pub value_col: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub kind: String,
    pub label: String,
    pub line: usize,
    pub col: usize,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Attribute(Attribute),
    Block(Block),
}

pub fn parse(tokens: &[Token]) -> Result<Vec<Statement>, Error> {
    Parser { tokens, pos: 0 }.statements(true)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn next(&mut self) -> Token {
        let t = self.peek().clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn fail(&self, expected: &str) -> Error {
        let t = self.peek();
        Error::new(t.line, t.col, format!("expected {expected}, found {}", t.kind.describe()))
    }

    fn statements(&mut self, top: bool) -> Result<Vec<Statement>, Error> {
        let mut out = Vec::new();
        loop {
            match self.peek().kind {
                TokenKind::Eof if top => return Ok(out),
                TokenKind::RBrace if !top => {
                    self.next();
                    return Ok(out);
                }
                TokenKind::Ident(_) => out.push(self.statement()?),
                _ if top => return Err(self.fail("identifier")),
                _ => return Err(self.fail("identifier or '}'")),
            }
        }
    }

    fn statement(&mut self) -> Result<Statement, Error> {
        let head = self.next();
        let TokenKind::Ident(name) = head.kind else {
            unreachable!()
        };
        match self.peek().kind.clone() {
            TokenKind::Equals => {
                self.next();
                let at = self.peek().clone();
                let value = self.value()?;
                Ok(Statement::Attribute(Attribute {
                    name,
                    line: head.line,
                    col: head.col,
                    value,
                    value_line: at.line,
                    value_col: at.col,
                }))
            }
            TokenKind::Str(label) => {
                self.next();
                if self.peek().kind != TokenKind::LBrace {
                    return Err(self.fail("'{'"));
                }
                self.next();
                let body = self.statements(false)?;
                Ok(Statement::Block(Block {
                    kind: name,
                    label,
                    line: head.line,
                    col: head.col,
                    body,
                }))
            }
            _ => Err(self.fail("'=' or block label")),
        }
    }

    fn value(&mut self) -> Result<Value, Error> {
        match self.peek().kind.clone() {
            TokenKind::Str(s) => {
                self.next();
                Ok(Value::Str(s))
            }
            TokenKind::LBracket => {
                self.next();
                self.list()
            }
            TokenKind::LBrace => {
                self.next();
                self.map()
            }
            _ => Err(self.fail("value")),
        }
    }

    fn string(&mut self) -> Result<String, Error> {
        match self.peek().kind.clone() {
            TokenKind::Str(s) => {
                self.next();
                Ok(s)
            }
            _ => Err(self.fail("string")),
        }
    }

    fn list(&mut self) -> Result<Value, Error> {
        let mut items = Vec::new();
        if self.peek().kind == TokenKind::RBracket {
            self.next();
            return Ok(Value::List(items));
        }
        loop {
            items.push(self.string()?);
            match self.peek().kind {
                TokenKind::Comma => {
                    self.next();
                    if self.peek().kind == TokenKind::RBracket {
                        self.next();
                        return Ok(Value::List(items));
                    }
                }
                TokenKind::RBracket => {
                    self.next();
                    return Ok(Value::List(items));
                }
                _ => return Err(self.fail("',' or ']'")),
            }
        }
    }

    fn map(&mut self) -> Result<Value, Error> {
        let mut entries = BTreeMap::new();
        loop {
            let key = match self.peek().kind.clone() {
                TokenKind::RBrace => {
                    self.next();
                    return Ok(Value::Map(entries));
                }
                TokenKind::Ident(k) => self.next_with(k),
                _ => return Err(self.fail("identifier or '}'")),
            };
            if self.peek().kind != TokenKind::Equals {
                return Err(self.fail("'='"));
            }
            self.next();
            let value = self.string()?;
            if entries.contains_key(&key.0) {
                return Err(Error::new(key.1, key.2, format!("duplicate key \"{}\"", key.0)));
            }
            entries.insert(key.0, value);
            if self.peek().kind == TokenKind::Comma {
                self.next();
            }
        }
    }

    fn next_with(&mut self, name: String) -> (String, usize, usize) {
        let t = self.next();
        (name, t.line, t.col)
    }
}
