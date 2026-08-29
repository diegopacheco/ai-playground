#[cfg(not(target_os = "macos"))]
compile_error!("monitor-stats reads display data from macOS CoreGraphics and only builds on macOS");

mod monitor;
mod quartz;
mod registry;
mod render;
mod vendor;

use std::process::ExitCode;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const HELP: &str = "\
🖥️  monitor-stats — every connected monitor, at a glance

USAGE
    monitor-stats [OPTIONS]

OPTIONS
    -j, --json       print the monitors as JSON
    -h, --help       print this help
    -V, --version    print the version
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => report(false),
        ["-h" | "--help"] => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        ["-V" | "--version"] => {
            println!("monitor-stats {VERSION}");
            ExitCode::SUCCESS
        }
        ["-j" | "--json"] => report(true),
        _ => {
            eprintln!("❌ unknown arguments: {}\n", args.join(" "));
            eprint!("{HELP}");
            ExitCode::FAILURE
        }
    }
}

fn report(as_json: bool) -> ExitCode {
    let monitors = monitor::detect();
    if as_json {
        print!("{}", render::json(&monitors));
    } else {
        print!("{}", render::text(&monitors));
    }
    if monitors.is_empty() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
