mod agents;
mod commenter;
mod compiler;
mod detect;
mod monitor;
mod pr;
mod server;
mod state;
mod tester;

use state::{AppState, PrInfo, SharedState};
use std::sync::{Arc, Mutex};

fn parse_duration(s: &str) -> Result<u64, String> {
    if s.ends_with('s') {
        s[..s.len() - 1]
            .parse::<u64>()
            .map_err(|_| format!("Invalid duration: {}", s))
    } else if s.ends_with('m') {
        s[..s.len() - 1]
            .parse::<u64>()
            .map(|m| m * 60)
            .map_err(|_| format!("Invalid duration: {}", s))
    } else {
        Err(format!("Invalid duration format: {}. Use e.g. 1s, 30s, 1m, 5m", s))
    }
}

fn get_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn print_usage() {
    println!("Usage: agent-pr [OPTIONS] <PR_URL>");
    println!();
    println!("OPTIONS:");
    println!("  --llm <agent>        LLM agent: claude, gemini, copilot, codex");
    println!("  --model <model>      Model for the LLM agent");
    println!("  --refresh <duration> Refresh interval: 1s, 30s, 1m, 5m (default: 5m)");
    println!("  --dry-run            Apply fixes locally but never push or comment on GitHub");
    println!("  --ui                 Open the web dashboard");
    println!("  --port <port>        Dashboard port (default: 3000, only with --ui)");
    println!();
    println!("Available agents and models:");
    for (name, models) in agents::available_agents() {
        println!("  {} ({})", name, models.join(", "));
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    let ui_mode = args.iter().any(|a| a == "--ui");
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let port: u16 = get_flag(&args, "--port")
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    let agent_name = get_flag(&args, "--llm").unwrap_or_else(|| "claude".to_string());
    let model = get_flag(&args, "--model").unwrap_or_else(|| "sonnet".to_string());

    let refresh_secs = match get_flag(&args, "--refresh") {
        Some(d) => match parse_duration(&d) {
            Ok(s) => s,
            Err(e) => {
                println!("Error: {}", e);
                return;
            }
        },
        None => 300,
    };

    let valid_agents: Vec<&str> = agents::available_agents().iter().map(|(n, _)| *n).collect();
    if !valid_agents.contains(&agent_name.as_str()) {
        println!("Error: unknown agent '{}'. Valid: {}", agent_name, valid_agents.join(", "));
        return;
    }

    let pr_url = args
        .iter()
        .filter(|a| !a.starts_with("--") && *a != &args[0])
        .filter(|a| {
            let prev_idx = args.iter().position(|x| x == *a).unwrap_or(0);
            prev_idx == 0
                || !["--llm", "--model", "--refresh", "--port"]
                    .contains(&args[prev_idx - 1].as_str())
        })
        .next()
        .cloned();

    let pr_url = match pr_url {
        Some(u) => u,
        None => {
            println!("Error: PR URL is required");
            print_usage();
            return;
        }
    };

    let (owner, repo, pr_number) = match pr::parse_pr_url(&pr_url) {
        Ok(parsed) => parsed,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };

    println!("Cloning PR #{} from {}/{}...", pr_number, owner, repo);
    let clone_path = match pr::clone_pr(&owner, &repo, pr_number) {
        Ok(path) => path,
        Err(e) => {
            println!("Error cloning: {}", e);
            return;
        }
    };

    let title = pr::get_pr_title(&owner, &repo, pr_number).unwrap_or_default();
    let branch = pr::get_pr_branch(&owner, &repo, pr_number).unwrap_or_default();
    let total_files = pr::count_files(&clone_path);

    let pr_info = PrInfo {
        url: pr_url,
        owner: owner.clone(),
        repo: repo.clone(),
        pr_number,
        title,
        branch,
        total_files,
        clone_path: clone_path.clone(),
        agent_name: agent_name.clone(),
        agent_model: model.clone(),
    };

    let state: SharedState = Arc::new(Mutex::new(AppState::new(pr_info)));

    {
        let mut st = state.lock().unwrap();
        st.dry_run = dry_run;
        st.refresh_file_tree();
    }

    if ui_mode {
        let server_state = state.clone();
        tokio::spawn(async move {
            server::start_server(server_state, port).await;
        });
        let _ = std::process::Command::new("open")
            .arg(format!("http://localhost:{}", port))
            .spawn();
        println!("Dashboard running at http://localhost:{}", port);
    }

    if dry_run {
        println!("DRY RUN mode enabled - no commits, pushes, or comments will be made on GitHub");
    }

    let refresh_label = if refresh_secs >= 60 {
        format!("{} minute(s)", refresh_secs / 60)
    } else {
        format!("{} second(s)", refresh_secs)
    };
    println!("Monitoring started. Checking every {}. Press Ctrl+C to stop.", refresh_label);

    let monitor_state = state.clone();
    let monitor_clone_path = clone_path.clone();
    let monitor_agent = agent_name.clone();
    let monitor_model = model.clone();
    let monitor_owner = owner.clone();
    let monitor_repo = repo.clone();
    tokio::spawn(async move {
        monitor::run_monitor_loop(
            &monitor_clone_path,
            &monitor_agent,
            &monitor_model,
            &monitor_owner,
            &monitor_repo,
            pr_number,
            refresh_secs,
            dry_run,
            monitor_state,
        )
        .await;
    });

    tokio::signal::ctrl_c().await.unwrap_or_default();
    println!("Shutting down.");
}
