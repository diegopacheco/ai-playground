mod agents;
mod cli;
mod cycle;
mod learning;
mod memory;
mod prompt;
mod repl;
mod review;
mod runner;

use agents::{is_valid_agent, get_default_model};
use cli::{get_base_dir, print_help, parse_args};
use cycle::{run_learning_cycles, print_summary};
use repl::run_repl;

#[tokio::main]
async fn main() {
    let args = parse_args();
    if args.show_help {
        print_help();
        return;
    }
    if !is_valid_agent(&args.agent) {
        eprintln!("Invalid agent: {}. Use: claude, codex, copilot, gemini", args.agent);
        return;
    }
    let base_dir = get_base_dir();
    let model = args.model.unwrap_or_else(|| get_default_model(&args.agent).to_string());
    if args.repl_mode || args.task.is_none() {
        if args.task.is_none() && !args.repl_mode {
            print_help();
            println!();
        }
        run_repl(&base_dir, &args.agent, &model, args.num_cycles).await;
        return;
    }
    let task = args.task.unwrap();
    println!("{}", "=".repeat(60));
    println!("Agent Learner Starting...");
    println!("Task: {}", task);
    println!("Agent: {} (use --agent to change)", args.agent);
    println!("Model: {} (use --model to change)", model);
    println!("Learning cycles: {}", args.num_cycles);
    println!("{}", "=".repeat(60));
    println!();
    let reports = run_learning_cycles(&base_dir, &task, &args.agent, &model, args.num_cycles).await;
    print_summary(&reports);
}
