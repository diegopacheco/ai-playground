use crossterm::style::Color;
use ram::catalog::{self, Item};
use ram::cli::{self, Action};
use ram::error::{Result, fail};
use ram::fetch::{self, GitSource, Source};
use ram::install::Installer;
use ram::kind::Target;
use ram::style::{paint, strong};
use ram::{banner, report, spinner, tui};
use std::path::PathBuf;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args).await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("   {} {}", strong("✘", Color::Red), paint(error, Color::Red));
            ExitCode::FAILURE
        }
    }
}

async fn run(args: &[String]) -> Result<ExitCode> {
    let cli = cli::parse(args)?;
    if cli.action == Action::Help {
        println!("{}", cli::USAGE);
        return Ok(ExitCode::SUCCESS);
    }
    let home = PathBuf::from(std::env::var("HOME").map_err(|_| "HOME is not set")?);
    banner::play().await?;
    let source = GitSource::new(&cli.repo, &home);
    let root = spinner::spin(&format!("fetching {}", source.label()), source.fetch()).await?;
    let revision = fetch::revision(&root).await.unwrap_or_default();
    let items = catalog::scan(&root)?;
    let label = format!("{} @ {}", source.label(), revision);
    println!("   {} {} items from {}", strong("✔", Color::Green), items.len(), paint(&label, Color::DarkGrey));
    let (chosen, targets) = match cli.action {
        Action::List => {
            report::list(&items);
            return Ok(ExitCode::SUCCESS);
        }
        Action::Install { selectors, targets } => (select(&items, &selectors)?, targets),
        _ => match pick(&items, &label)? {
            Some(pair) => pair,
            None => {
                println!("   {}", paint("nothing installed, bye", Color::DarkGrey));
                return Ok(ExitCode::SUCCESS);
            }
        },
    };
    let installer = Installer { home: home.clone(), marketplace_source: source.marketplace_source() };
    let count = chosen.len();
    let outcomes = spinner::spin(&format!("installing {count} items"), installer.install_all(chosen, &targets)).await;
    let failed = report::outcomes(&outcomes, &home);
    Ok(if failed == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE })
}

fn pick(items: &[Item], label: &str) -> Result<Option<(Vec<Item>, Vec<Target>)>> {
    let Some(targets) = tui::choose_targets(label, items.len())? else {
        return Ok(None);
    };
    let available: Vec<Item> = items.iter().filter(|i| targets.iter().any(|t| t.supports(i.kind))).cloned().collect();
    let Some(indexes) = tui::choose_items(&available, &targets)? else {
        return Ok(None);
    };
    Ok(Some((indexes.into_iter().map(|i| available[i].clone()).collect(), targets)))
}

fn select(items: &[Item], selectors: &[String]) -> Result<Vec<Item>> {
    let mut chosen = Vec::new();
    for selector in selectors {
        let matched: Vec<&Item> = items.iter().filter(|i| i.matches(selector)).collect();
        if matched.is_empty() {
            return fail(format!("nothing named {selector}, try ram list"));
        }
        chosen.extend(matched.into_iter().cloned());
    }
    Ok(chosen)
}
