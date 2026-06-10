mod agent;
mod json;
mod render;
mod term;

use agent::Cache;
use render::{Sort, State};
use std::process::Command;
use std::time::{Duration, Instant};

fn clock() -> String {
    Command::new("date")
        .arg("+%H:%M:%S")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn host() -> String {
    Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "local".into())
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "--help" | "-h" => {
            println!("agentop - top for claude code agents");
            println!();
            println!("usage: agentop [--once | --html | --help]");
            println!("  (no args)  live TUI monitor");
            println!("  --once     print a single frame and exit");
            println!("  --html     print one frame as an HTML snapshot");
            println!();
            println!("keys: q quit  s sort  +/- refresh speed");
        }
        "--once" => snapshot(false),
        "--html" => snapshot(true),
        _ => live(),
    }
}

fn snapshot(as_html: bool) {
    let mut cache = Cache::new();
    let mut agents = agent::collect(&mut cache);
    let state = State { sort: Sort::Cpu, interval_ms: 2000 };
    render::sort_agents(&mut agents, state.sort);
    let w = 116;
    let frame = render::frame(&agents, &state, w, 0, agent::now_ms(), &clock(), &host());
    if as_html {
        print!("{}", render::html(&frame));
    } else {
        for line in &frame {
            println!("{}", term::ansi_line(line, w));
        }
    }
}

fn live() {
    let t = term::Term::enter();
    let host = host();
    let mut cache = Cache::new();
    let mut state = State { sort: Sort::Cpu, interval_ms: 2000 };
    let mut last = Instant::now() - Duration::from_secs(3600);
    let mut dirty = true;

    loop {
        for b in term::poll_keys() {
            match b {
                b'q' | b'Q' | 3 => return,
                b's' | b'S' => {
                    state.sort = next_sort(state.sort);
                    dirty = true;
                }
                b'+' | b'=' => {
                    state.interval_ms = (state.interval_ms.saturating_sub(1000)).max(1000);
                    dirty = true;
                }
                b'-' | b'_' => {
                    state.interval_ms = (state.interval_ms + 1000).min(10000);
                    dirty = true;
                }
                _ => {}
            }
        }

        if dirty || last.elapsed() >= Duration::from_millis(state.interval_ms) {
            let (w, h) = t.size();
            let mut agents = agent::collect(&mut cache);
            render::sort_agents(&mut agents, state.sort);
            let frame = render::frame(&agents, &state, w, h, agent::now_ms(), &clock(), &host);
            t.draw(&frame, w);
            last = Instant::now();
            dirty = false;
        }

        std::thread::sleep(Duration::from_millis(80));
    }
}

fn next_sort(s: Sort) -> Sort {
    match s {
        Sort::Cpu => Sort::Ctx,
        Sort::Ctx => Sort::Up,
        Sort::Up => Sort::Status,
        Sort::Status => Sort::Pid,
        Sort::Pid => Sort::Cpu,
    }
}
