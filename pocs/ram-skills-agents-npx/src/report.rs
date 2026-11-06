use crate::catalog::Item;
use crate::install::Outcome;
use crate::kind::Kind;
use crate::style::{bold, paint, strong};
use crossterm::style::Color;
use std::path::Path;

pub fn tilde(path: &str, home: &Path) -> String {
    let home = home.to_string_lossy();
    match path.strip_prefix(home.as_ref()) {
        Some(rest) => format!("~{rest}"),
        None => path.to_string(),
    }
}

pub fn list(items: &[Item]) {
    for kind in Kind::ALL {
        let group: Vec<&Item> = items.iter().filter(|i| i.kind == kind).collect();
        println!("\n   {} {}", bold(format!("{}s", kind.label())), paint(format!("({})", group.len()), Color::DarkGrey));
        for item in group {
            println!("     {:<28} {}", item.name, paint(&item.path, Color::DarkGrey));
        }
    }
    println!();
}

pub fn outcomes(outcomes: &[Outcome], home: &Path) -> usize {
    let mut failed = 0;
    for outcome in outcomes {
        let head = format!("{:<7} {:<28} {:<6}", outcome.item.kind.label(), outcome.item.name, outcome.target.label());
        match &outcome.result {
            Ok(dest) => println!("   {} {} {} {}", strong("✔", Color::Green), head, paint("→", Color::DarkGrey), paint(tilde(dest, home), Color::DarkGrey)),
            Err(error) => {
                failed += 1;
                eprintln!("   {} {} {}", strong("✘", Color::Red), head, paint(error, Color::Red));
            }
        }
    }
    let ok = outcomes.len() - failed;
    println!();
    if failed == 0 {
        println!("   {}", strong(format!("✨ {ok} installed"), Color::Green));
    } else {
        eprintln!("   {} installed, {} failed", strong(ok, Color::Green), strong(failed, Color::Red));
    }
    failed
}
