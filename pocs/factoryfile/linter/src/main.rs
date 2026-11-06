use std::path::PathBuf;
use std::process::ExitCode;

use factoryfile_lint::{Level, lint_file};

fn main() -> ExitCode {
    let mut paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        paths.push(PathBuf::from("Factoryfile"));
    }
    let (mut errors, mut warnings) = (0, 0);
    for path in &paths {
        let shown = path.display().to_string();
        match lint_file(path) {
            Ok(diagnostics) => {
                for d in &diagnostics {
                    println!("{}", d.render(&shown));
                    match d.level {
                        Level::Error => errors += 1,
                        Level::Warning => warnings += 1,
                    }
                }
            }
            Err(e) => {
                eprintln!("{shown}: cannot read: {e}");
                errors += 1;
            }
        }
    }
    println!("{} file(s), {errors} error(s), {warnings} warning(s)", paths.len());
    if errors > 0 { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
