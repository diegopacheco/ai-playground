use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use factory_cli::args::{self, Args, Mode};
use factory_cli::{html, terminal};
use factoryfile::{Factory, Factoryfile};

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    match args::parse(&raw).and_then(|a| run(&a)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("factory-cli: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    if args.mode == Mode::Help {
        println!("{}", args::USAGE);
        return Ok(());
    }
    let file = factoryfile::parse_file(&args.path).map_err(|e| format!("{}: {e}", args.path.display()))?;
    let factories = select(&file, args.factory.as_deref())?;
    match args.mode {
        Mode::Render => {
            let drawn: Vec<String> = factories.iter().map(|f| terminal::render(f, args.color)).collect();
            print!("{}", drawn.join("\n\n"));
            Ok(())
        }
        _ => render_ui(args, &factories),
    }
}

fn select<'a>(file: &'a Factoryfile, name: Option<&str>) -> Result<Vec<&'a Factory>, String> {
    let chosen: Vec<&Factory> = file.factories.iter().filter(|f| name.is_none_or(|n| f.name == n)).collect();
    match (chosen.is_empty(), name) {
        (true, Some(n)) => Err(format!("no factory named \"{n}\"")),
        (true, None) => Err("the file declares no factory".to_string()),
        _ => Ok(chosen),
    }
}

fn render_ui(args: &Args, factories: &[&Factory]) -> Result<(), String> {
    let stem = args.path.file_stem().and_then(|s| s.to_str()).unwrap_or("factory").to_string();
    let out = args.out.clone().unwrap_or_else(|| PathBuf::from("out").join(format!("{stem}.html")));
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&out, html::page(&stem, factories)).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("wrote {}", out.display());
    if args.open {
        open(&out)?;
    }
    Ok(())
}

fn open(path: &Path) -> Result<(), String> {
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    Command::new(opener).arg(path).status().map_err(|e| format!("{opener}: {e}"))?;
    Ok(())
}
