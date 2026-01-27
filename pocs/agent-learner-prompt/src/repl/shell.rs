use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::agents;
use crate::memory::{show_memory, show_mistakes};
use crate::prompt::show_prompts;
use crate::cycle::{run_learning_cycles, print_summary, sanitize_project_name, get_solutions_dir};

pub fn print_repl_help() {
    println!("REPL Commands:");
    println!("  :quit, :q               Exit the REPL");
    println!("  :agent <NAME>           Switch agent (claude, codex, copilot, gemini)");
    println!("  :agent                  Show current agent");
    println!("  :model <NAME>           Switch model");
    println!("  :model                  Show current model");
    println!("  :cycles <N>             Set number of cycles (e.g. :cycles 5)");
    println!("  :cycles                 Show current cycles");
    println!("  :memory, :m             Show current learnings");
    println!("  :mistakes               Show current mistakes");
    println!("  :prompts, :p            Show prompt history");
    println!("  :clear                  Clear screen");
    println!("  :help, :h               Show this help");
    println!();
    println!("Enter any task to start a learning session.");
}

pub async fn run_repl(base_dir: &Path, initial_agent: &str, initial_model: &str, initial_cycles: u32) {
    let mut agent = initial_agent.to_string();
    let mut model = initial_model.to_string();
    let mut num_cycles = initial_cycles;
    let mut last_project_dir: Option<PathBuf> = None;
    println!("Agent Learner REPL - Interactive Mode");
    println!("Type :help for commands, or enter a task to start learning");
    println!("Agent: {} | Model: {} | Cycles: {}", agent, model, num_cycles);
    println!();
    loop {
        print!("agent> ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == ":quit" || input == ":q" {
            println!("Goodbye!");
            break;
        } else if input == ":memory" || input == ":m" {
            if let Some(ref dir) = last_project_dir {
                show_memory(dir);
            } else {
                println!("No project yet. Run a task first.");
            }
        } else if input == ":mistakes" {
            if let Some(ref dir) = last_project_dir {
                show_mistakes(dir);
            } else {
                println!("No project yet. Run a task first.");
            }
        } else if input == ":prompts" || input == ":p" {
            if let Some(ref dir) = last_project_dir {
                show_prompts(dir);
            } else {
                println!("No project yet. Run a task first.");
            }
        } else if input == ":help" || input == ":h" {
            print_repl_help();
        } else if input == ":clear" {
            print!("\x1B[2J\x1B[1;1H");
            let _ = io::stdout().flush();
        } else if input.starts_with(":agent") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 2 {
                let new_agent = parts[1];
                if agents::is_valid_agent(new_agent) {
                    agent = new_agent.to_string();
                    model = agents::get_default_model(&agent).to_string();
                    println!("Agent set to: {} (model: {})", agent, model);
                } else {
                    println!("Invalid agent: {}. Use: claude, codex, copilot, gemini", new_agent);
                }
            } else {
                println!("Current agent: {}", agent);
                println!("Usage: :agent <name>");
            }
        } else if input.starts_with(":model") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 2 {
                model = parts[1].to_string();
                println!("Model set to: {}", model);
            } else {
                println!("Current model: {}", model);
                println!("Usage: :model <name>");
            }
        } else if input.starts_with(":cycles") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(n) = parts[1].parse::<u32>() {
                    if n > 0 && n <= 10 {
                        num_cycles = n;
                        println!("Cycles set to {}", num_cycles);
                    } else {
                        println!("Cycles must be between 1 and 10");
                    }
                } else {
                    println!("Invalid number: {}", parts[1]);
                }
            } else {
                println!("Current cycles: {}", num_cycles);
                println!("Usage: :cycles <N>");
            }
        } else if input.starts_with(':') {
            println!("Unknown command: {}", input);
            println!("Type :help for available commands");
        } else {
            println!("\nStarting learning session: {}", input);
            let project_name = sanitize_project_name(input);
            last_project_dir = Some(base_dir.join(get_solutions_dir()).join(&project_name));
            let reports = run_learning_cycles(base_dir, input, &agent, &model, num_cycles).await;
            print_summary(&reports);
            println!();
        }
    }
}
