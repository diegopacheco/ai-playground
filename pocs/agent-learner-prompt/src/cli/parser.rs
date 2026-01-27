use std::env;
use std::path::PathBuf;

pub const DEFAULT_CYCLES: u32 = 3;
pub const DEFAULT_AGENT: &str = "claude";

pub fn get_base_dir() -> PathBuf {
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn print_help() {
    println!("Agent Learner - Self Learning Code Generation Agent");
    println!();
    println!("Usage: agent-learner [OPTIONS] [TASK]");
    println!();
    println!("Arguments:");
    println!("  [TASK]                  The task description for code generation");
    println!("                          If not provided, enters REPL mode");
    println!();
    println!("Options:");
    println!("  --agent <AGENT>, -a     CLI agent to use (claude, codex, copilot, gemini)");
    println!("  --model <MODEL>, -m     Model to use for the agent");
    println!("  --cycles <N>, -c        Number of learning cycles (default: 3)");
    println!("  --repl                  Enter interactive REPL mode");
    println!("  --help                  Show this help message");
    println!();
    println!("Supported Agents:");
    println!("  claude   - Claude CLI (default)");
    println!("  codex    - OpenAI Codex CLI");
    println!("  copilot  - GitHub Copilot CLI");
    println!("  gemini   - Google Gemini CLI");
    println!();
    println!("REPL Commands:");
    println!("  :quit, :q               Exit the REPL");
    println!("  :agent <NAME>           Switch agent (claude, codex, copilot, gemini)");
    println!("  :model <NAME>           Switch model");
    println!("  :cycles <N>             Set number of cycles (e.g. :cycles 5)");
    println!("  :memory, :m             Show current learnings");
    println!("  :mistakes               Show current mistakes");
    println!("  :prompts, :p            Show prompt history");
    println!("  :help, :h               Show REPL help");
}

pub struct Args {
    pub agent: String,
    pub model: Option<String>,
    pub task: Option<String>,
    pub repl_mode: bool,
    pub num_cycles: u32,
    pub show_help: bool,
}

pub fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut agent = DEFAULT_AGENT.to_string();
    let mut model: Option<String> = None;
    let mut task: Option<String> = None;
    let mut repl_mode = false;
    let mut num_cycles = DEFAULT_CYCLES;
    let mut show_help = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                show_help = true;
            }
            "--repl" => {
                repl_mode = true;
            }
            "--agent" | "-a" => {
                if i + 1 < args.len() {
                    agent = args[i + 1].clone();
                    i += 1;
                }
            }
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--cycles" | "-c" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<u32>() {
                        if n > 0 && n <= 10 {
                            num_cycles = n;
                        }
                    }
                    i += 1;
                }
            }
            _ => {
                if !args[i].starts_with('-') {
                    task = Some(args[i..].join(" "));
                    break;
                }
            }
        }
        i += 1;
    }
    Args {
        agent,
        model,
        task,
        repl_mode,
        num_cycles,
        show_help,
    }
}
