pub mod diagnostic;
pub mod errors;
pub mod warnings;

use std::path::Path;

pub use diagnostic::{Diagnostic, Level};

pub fn lint(src: &str, base: &Path) -> Vec<Diagnostic> {
    let statements = match factoryfile::parse_statements(src) {
        Ok(s) => s,
        Err(e) => return vec![errors::from_syntax(&e)],
    };
    let (_, issues) = factoryfile::validate(&statements);
    let mut out: Vec<Diagnostic> = issues.iter().map(errors::from_issue).collect();
    out.extend(warnings::check(&statements, base));
    out.sort_by_key(|d| (d.line, d.col, d.level));
    out
}

pub fn lint_file(path: &Path) -> Result<Vec<Diagnostic>, std::io::Error> {
    let src = std::fs::read_to_string(path)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(lint(&src, base))
}
