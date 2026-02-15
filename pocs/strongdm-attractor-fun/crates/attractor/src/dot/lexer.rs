#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Digraph,
    Graph,
    Subgraph,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Arrow,
    Dash,
    Semi,
    Comma,
    Eq,
    Ident(String),
    QuotedString(String),
    Eof,
}

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied();
        self.pos += 1;
        ch
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.peek().map_or(false, |c| c.is_whitespace()) {
                self.advance();
            }
            if self.peek() == Some('/') && self.input.get(self.pos + 1) == Some(&'/') {
                while self.peek().map_or(false, |c| c != '\n') {
                    self.advance();
                }
                continue;
            }
            if self.peek() == Some('/') && self.input.get(self.pos + 1) == Some(&'*') {
                self.advance();
                self.advance();
                loop {
                    if self.peek().is_none() {
                        break;
                    }
                    if self.peek() == Some('*') && self.input.get(self.pos + 1) == Some(&'/') {
                        self.advance();
                        self.advance();
                        break;
                    }
                    self.advance();
                }
                continue;
            }
            break;
        }
    }

    fn read_string(&mut self) -> String {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('\\') => {
                    if let Some(c) = self.advance() {
                        s.push(c);
                    }
                }
                Some('"') => break,
                Some(c) => s.push(c),
                None => break,
            }
        }
        s
    }

    fn read_ident(&mut self, first: char) -> String {
        let mut s = String::new();
        s.push(first);
        while self
            .peek()
            .map_or(false, |c| c.is_alphanumeric() || c == '_' || c == '.')
        {
            s.push(self.advance().unwrap());
        }
        s
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();
        match self.advance() {
            None => Token::Eof,
            Some('{') => Token::LBrace,
            Some('}') => Token::RBrace,
            Some('[') => Token::LBracket,
            Some(']') => Token::RBracket,
            Some(';') => Token::Semi,
            Some(',') => Token::Comma,
            Some('=') => Token::Eq,
            Some('-') => {
                if self.peek() == Some('>') {
                    self.advance();
                    Token::Arrow
                } else if self.peek() == Some('-') {
                    self.advance();
                    Token::Dash
                } else {
                    Token::Dash
                }
            }
            Some('"') => Token::QuotedString(self.read_string()),
            Some(c) if c.is_alphanumeric() || c == '_' => {
                let ident = self.read_ident(c);
                match ident.as_str() {
                    "digraph" => Token::Digraph,
                    "graph" => Token::Graph,
                    "subgraph" => Token::Subgraph,
                    _ => Token::Ident(ident),
                }
            }
            Some(c) => Token::Ident(c.to_string()),
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            if tok == Token::Eof {
                tokens.push(tok);
                break;
            }
            tokens.push(tok);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lex = Lexer::new("digraph G { a -> b; }");
        let tokens = lex.tokenize();
        assert_eq!(tokens[0], Token::Digraph);
        assert_eq!(tokens[1], Token::Ident("G".into()));
        assert_eq!(tokens[2], Token::LBrace);
        assert_eq!(tokens[3], Token::Ident("a".into()));
        assert_eq!(tokens[4], Token::Arrow);
        assert_eq!(tokens[5], Token::Ident("b".into()));
        assert_eq!(tokens[6], Token::Semi);
        assert_eq!(tokens[7], Token::RBrace);
    }

    #[test]
    fn test_quoted_strings() {
        let mut lex = Lexer::new(r#"a [label="hello world"]"#);
        let tokens = lex.tokenize();
        assert_eq!(tokens[2], Token::Ident("label".into()));
        assert_eq!(tokens[3], Token::Eq);
        assert_eq!(tokens[4], Token::QuotedString("hello world".into()));
    }
}
