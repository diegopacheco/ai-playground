pub mod error;
pub mod lexer;
pub mod model;
pub mod parser;
pub mod schema;

use std::path::Path;

pub use error::{Error, LoadError};
pub use model::{Agent, Factory, Factoryfile, Loop, Stage};
pub use parser::{Attribute, Block, Statement, Value};
pub use schema::{Issue, IssueKind};

pub fn parse_statements(src: &str) -> Result<Vec<Statement>, Error> {
    let tokens = lexer::tokenize(src)?;
    parser::parse(&tokens)
}

pub fn validate(statements: &[Statement]) -> (Factoryfile, Vec<Issue>) {
    schema::validate(statements)
}

pub fn parse(src: &str) -> Result<Factoryfile, Error> {
    let statements = parse_statements(src)?;
    let (file, issues) = validate(&statements);
    match issues.into_iter().next() {
        Some(issue) => Err(issue.error),
        None => Ok(file),
    }
}

pub fn parse_file(path: impl AsRef<Path>) -> Result<Factoryfile, LoadError> {
    let src = std::fs::read_to_string(path)?;
    Ok(parse(&src)?)
}
