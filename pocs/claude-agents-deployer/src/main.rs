use dialoguer::{theme::ColorfulTheme, Confirm, MultiSelect, Select};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

struct Agent {
    name: String,
    filename: String,
    path: PathBuf,
}

fn discover_agents(agents_dir: &Path) -> Vec<Agent> {
    let mut agents = Vec::new();
    if !agents_dir.exists() {
        return agents;
    }
    for entry in WalkDir::new(agents_dir).max_depth(1).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "md") {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                let name = filename
                    .trim_end_matches(".md")
                    .replace("-", " ")
                    .replace("agent", "")
                    .trim()
                    .split_whitespace()
                    .map(|word| {
                        let mut chars = word.chars();
                        match chars.next() {
                            None => String::new(),
                            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                agents.push(Agent {
                    name,
                    filename: filename.to_string(),
                    path: path.to_path_buf(),
                });
            }
        }
    }
    agents.sort_by(|a, b| a.name.cmp(&b.name));
    agents
}

fn get_claude_agents_dir(global: bool) -> PathBuf {
    if global {
        dirs::home_dir()
            .expect("Could not find home directory")
            .join(".claude")
            .join("agents")
    } else {
        PathBuf::from(".claude").join("agents")
    }
}

fn get_claude_commands_dir(global: bool) -> PathBuf {
    if global {
        dirs::home_dir()
            .expect("Could not find home directory")
            .join(".claude")
            .join("commands")
    } else {
        PathBuf::from(".claude").join("commands")
    }
}

fn install_agent(agent: &Agent, target_dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(target_dir)?;
    let dest = target_dir.join(&agent.filename);
    fs::copy(&agent.path, &dest)?;
    Ok(())
}

fn create_command(agent: &Agent, commands_dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(commands_dir)?;
    let command_name = agent.filename.trim_end_matches(".md");
    let command_file = commands_dir.join(format!("{}.md", command_name));
    let content = format!(
        "# {}\n\nInvoke the {} agent.\n\nUsage: /{}\n",
        agent.name, agent.name, command_name
    );
    fs::write(&command_file, content)?;
    Ok(())
}

fn main() {
    let theme = ColorfulTheme::default();
    println!("\n  Claude Code Agent Deployer\n");
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let agents_dir = if exe_dir.join("agents").exists() {
        exe_dir.join("agents")
    } else {
        PathBuf::from("agents")
    };
    let agents = discover_agents(&agents_dir);
    if agents.is_empty() {
        println!("No agents found in {:?}", agents_dir);
        println!("Add .md files to the agents/ folder and try again.");
        return;
    }
    println!("Found {} agents:\n", agents.len());
    for agent in &agents {
        println!("  - {}", agent.name);
    }
    println!();
    let install_all = Confirm::with_theme(&theme)
        .with_prompt("Install all agents?")
        .default(true)
        .interact()
        .unwrap();
    let selected_indices: Vec<usize> = if install_all {
        (0..agents.len()).collect()
    } else {
        let agent_names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        MultiSelect::with_theme(&theme)
            .with_prompt("Select agents to install (space to toggle, enter to confirm)")
            .items(&agent_names)
            .interact()
            .unwrap()
    };
    if selected_indices.is_empty() {
        println!("No agents selected. Exiting.");
        return;
    }
    let install_options = vec!["Global (~/.claude/)", "Local (./.claude/)"];
    let install_choice = Select::with_theme(&theme)
        .with_prompt("Installation type")
        .items(&install_options)
        .default(0)
        .interact()
        .unwrap();
    let global = install_choice == 0;
    let agents_target = get_claude_agents_dir(global);
    let commands_target = get_claude_commands_dir(global);
    let selected_agents: Vec<&Agent> = selected_indices.iter().map(|&i| &agents[i]).collect();
    let selected_names: Vec<&str> = selected_agents.iter().map(|a| a.name.as_str()).collect();
    let command_indices = MultiSelect::with_theme(&theme)
        .with_prompt("Turn these agents into commands? (space to toggle, enter to confirm)")
        .items(&selected_names)
        .interact()
        .unwrap();
    println!("\nInstalling agents...\n");
    let mut installed_count = 0;
    let mut command_count = 0;
    for (idx, agent) in selected_agents.iter().enumerate() {
        match install_agent(agent, &agents_target) {
            Ok(_) => {
                println!("  Installed: {}", agent.name);
                installed_count += 1;
            }
            Err(e) => println!("  Failed to install {}: {}", agent.name, e),
        }
        if command_indices.contains(&idx) {
            match create_command(agent, &commands_target) {
                Ok(_) => {
                    println!("  Created command: /{}", agent.filename.trim_end_matches(".md"));
                    command_count += 1;
                }
                Err(e) => println!("  Failed to create command for {}: {}", agent.name, e),
            }
        }
    }
    println!("\nDone!");
    println!("  Installed {} agents to {:?}", installed_count, agents_target);
    if command_count > 0 {
        println!("  Created {} commands in {:?}", command_count, commands_target);
    }
}
