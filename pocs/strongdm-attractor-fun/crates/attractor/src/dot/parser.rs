use super::ast::*;
use super::lexer::{Lexer, Token};
use std::collections::HashMap;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer::new(input);
        Self {
            tokens: lexer.tokenize(),
            pos: 0,
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        let tok = self.advance();
        if std::mem::discriminant(&tok) == std::mem::discriminant(expected) {
            Ok(())
        } else {
            Err(format!("expected {:?}, got {:?}", expected, tok))
        }
    }

    fn ident_or_string(&mut self) -> Result<String, String> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            Token::QuotedString(s) => Ok(s),
            tok => Err(format!("expected identifier, got {:?}", tok)),
        }
    }

    fn parse_attrs(&mut self) -> Result<HashMap<String, String>, String> {
        let mut attrs = HashMap::new();
        if self.peek() != &Token::LBracket {
            return Ok(attrs);
        }
        self.advance();
        loop {
            if self.peek() == &Token::RBracket {
                self.advance();
                break;
            }
            let key = self.ident_or_string()?;
            self.expect(&Token::Eq)?;
            let value = self.ident_or_string()?;
            attrs.insert(key, value);
            if self.peek() == &Token::Comma || self.peek() == &Token::Semi {
                self.advance();
            }
        }
        Ok(attrs)
    }

    pub fn parse(&mut self) -> Result<DotGraph, String> {
        let is_digraph = match self.advance() {
            Token::Digraph => true,
            Token::Graph => false,
            tok => return Err(format!("expected digraph/graph, got {:?}", tok)),
        };
        let name = match self.peek() {
            Token::Ident(_) | Token::QuotedString(_) => self.ident_or_string()?,
            _ => String::new(),
        };
        self.expect(&Token::LBrace)?;
        let mut graph = DotGraph::new(name, is_digraph);
        self.parse_body(&mut graph)?;
        self.expect(&Token::RBrace)?;
        Ok(graph)
    }

    fn parse_body(&mut self, graph: &mut DotGraph) -> Result<(), String> {
        loop {
            match self.peek() {
                Token::RBrace | Token::Eof => break,
                Token::Subgraph => {
                    self.advance();
                    let sg_name = match self.peek() {
                        Token::Ident(_) | Token::QuotedString(_) => self.ident_or_string()?,
                        _ => String::new(),
                    };
                    self.expect(&Token::LBrace)?;
                    let mut subgraph = DotSubgraph {
                        name: sg_name,
                        nodes: Vec::new(),
                        edges: Vec::new(),
                    };
                    self.parse_subgraph_body(&mut subgraph)?;
                    self.expect(&Token::RBrace)?;
                    graph.subgraphs.push(subgraph);
                    if self.peek() == &Token::Semi {
                        self.advance();
                    }
                }
                Token::Ident(_) | Token::QuotedString(_) => {
                    let id = self.ident_or_string()?;
                    if id == "node" || id == "edge" || id == "graph" {
                        let _attrs = self.parse_attrs()?;
                        if self.peek() == &Token::Semi {
                            self.advance();
                        }
                        continue;
                    }
                    match self.peek() {
                        Token::Arrow | Token::Dash => {
                            self.advance();
                            let to = self.ident_or_string()?;
                            let attrs = self.parse_attrs()?;
                            graph.edges.push(DotEdge {
                                from: id,
                                to,
                                attrs,
                            });
                        }
                        Token::LBracket => {
                            let attrs = self.parse_attrs()?;
                            graph.nodes.push(DotNode { id, attrs });
                        }
                        _ => {
                            graph.nodes.push(DotNode {
                                id,
                                attrs: HashMap::new(),
                            });
                        }
                    }
                    if self.peek() == &Token::Semi {
                        self.advance();
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }
        Ok(())
    }

    fn parse_subgraph_body(&mut self, sg: &mut DotSubgraph) -> Result<(), String> {
        loop {
            match self.peek() {
                Token::RBrace | Token::Eof => break,
                Token::Ident(_) | Token::QuotedString(_) => {
                    let id = self.ident_or_string()?;
                    match self.peek() {
                        Token::Arrow | Token::Dash => {
                            self.advance();
                            let to = self.ident_or_string()?;
                            let attrs = self.parse_attrs()?;
                            sg.edges.push(DotEdge {
                                from: id,
                                to,
                                attrs,
                            });
                        }
                        Token::LBracket => {
                            let attrs = self.parse_attrs()?;
                            sg.nodes.push(DotNode { id, attrs });
                        }
                        _ => {
                            sg.nodes.push(DotNode {
                                id,
                                attrs: HashMap::new(),
                            });
                        }
                    }
                    if self.peek() == &Token::Semi {
                        self.advance();
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_digraph() {
        let input = r#"digraph pipeline {
            start [type="llm" prompt="hello"];
            process [type="transform" operation="json_extract"];
            start -> process;
        }"#;
        let mut parser = Parser::new(input);
        let graph = parser.parse().unwrap();
        assert_eq!(graph.name, "pipeline");
        assert!(graph.is_digraph);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].from, "start");
        assert_eq!(graph.edges[0].to, "process");
    }

    #[test]
    fn test_parse_node_attrs() {
        let input = r#"digraph G {
            n1 [type="llm", model="gpt-4o", prompt="test"];
        }"#;
        let mut parser = Parser::new(input);
        let graph = parser.parse().unwrap();
        assert_eq!(graph.nodes[0].attrs["type"], "llm");
        assert_eq!(graph.nodes[0].attrs["model"], "gpt-4o");
    }

    #[test]
    fn test_parse_subgraph() {
        let input = r#"digraph G {
            subgraph cluster_0 {
                a; b;
                a -> b;
            }
        }"#;
        let mut parser = Parser::new(input);
        let graph = parser.parse().unwrap();
        assert_eq!(graph.subgraphs.len(), 1);
        assert_eq!(graph.subgraphs[0].name, "cluster_0");
        assert_eq!(graph.subgraphs[0].nodes.len(), 2);
        assert_eq!(graph.subgraphs[0].edges.len(), 1);
    }
}
