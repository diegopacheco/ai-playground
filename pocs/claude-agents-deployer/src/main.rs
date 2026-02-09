use dialoguer::{theme::ColorfulTheme, Confirm, MultiSelect, Select};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Platform {
    Claude,
    Codex,
}

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

fn get_claude_skills_dir(global: bool) -> PathBuf {
    if global {
        dirs::home_dir()
            .expect("Could not find home directory")
            .join(".claude")
            .join("skills")
    } else {
        PathBuf::from(".claude").join("skills")
    }
}

fn get_codex_prompts_dir(global: bool) -> PathBuf {
    if global {
        dirs::home_dir()
            .expect("Could not find home directory")
            .join(".codex")
            .join("prompts")
    } else {
        PathBuf::from(".codex").join("prompts")
    }
}

fn get_codex_skills_dir(global: bool) -> PathBuf {
    if global {
        dirs::home_dir()
            .expect("Could not find home directory")
            .join(".codex")
            .join("skills")
    } else {
        PathBuf::from(".codex").join("skills")
    }
}

fn install_workflow_for_claude(skills_dir: &Path, commands_dir: &Path, exe_dir: &Path) -> std::io::Result<()> {
    let skill_source = if exe_dir.join("skills").join("workflow-skill").join("SKILL.md").exists() {
        exe_dir.join("skills").join("workflow-skill").join("SKILL.md")
    } else {
        PathBuf::from("skills").join("workflow-skill").join("SKILL.md")
    };
    let command_source = if exe_dir.join("skills").join("workflow.md").exists() {
        exe_dir.join("skills").join("workflow.md")
    } else {
        PathBuf::from("skills").join("workflow.md")
    };
    let skill_target = skills_dir.join("workflow-skill");
    fs::create_dir_all(&skill_target)?;
    fs::copy(&skill_source, skill_target.join("SKILL.md"))?;
    let command_target = commands_dir.join("ad");
    fs::create_dir_all(&command_target)?;
    fs::copy(&command_source, command_target.join("wf.md"))?;
    Ok(())
}

fn install_workflow_for_codex(skills_dir: &Path, prompts_dir: &Path, exe_dir: &Path) -> std::io::Result<()> {
    let skill_source = if exe_dir.join("skills").join("workflow-skill").join("SKILL.md").exists() {
        exe_dir.join("skills").join("workflow-skill").join("SKILL.md")
    } else {
        PathBuf::from("skills").join("workflow-skill").join("SKILL.md")
    };
    let prompt_source = if exe_dir.join("skills").join("workflow.md").exists() {
        exe_dir.join("skills").join("workflow.md")
    } else {
        PathBuf::from("skills").join("workflow.md")
    };
    let skill_target = skills_dir.join("workflow-skill");
    fs::create_dir_all(&skill_target)?;
    fs::copy(&skill_source, skill_target.join("SKILL.md"))?;
    fs::create_dir_all(prompts_dir)?;
    fs::copy(&prompt_source, prompts_dir.join("workflow.md"))?;
    let ad_target = prompts_dir.join("ad");
    fs::create_dir_all(&ad_target)?;
    fs::copy(&prompt_source, ad_target.join("wf.md"))?;
    Ok(())
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
    println!("\n  Claude/Codex Agent Deployer\n");
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let platform_options = vec!["Claude", "Codex"];
    let platform_indices = MultiSelect::with_theme(&theme)
        .with_prompt("Select target platform(s) (space to toggle, enter to confirm)")
        .items(&platform_options)
        .defaults(&[true, true])
        .interact()
        .unwrap();
    if platform_indices.is_empty() {
        println!("No platform selected. Exiting.");
        return;
    }
    let selected_platforms: Vec<Platform> = platform_indices
        .iter()
        .map(|i| match i {
            0 => Platform::Claude,
            1 => Platform::Codex,
            _ => unreachable!(),
        })
        .collect();

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
    let selected_agents: Vec<&Agent> = selected_indices.iter().map(|&i| &agents[i]).collect();
    let selected_names: Vec<&str> = selected_agents.iter().map(|a| a.name.as_str()).collect();
    let has_claude = selected_platforms.contains(&Platform::Claude);
    let has_codex = selected_platforms.contains(&Platform::Codex);
    let claude_global = if has_claude {
        let install_options = vec!["Global (~/.claude/)", "Local (./.claude/)"];
        let install_choice = Select::with_theme(&theme)
            .with_prompt("Claude installation type")
            .items(&install_options)
            .default(0)
            .interact()
            .unwrap();
        install_choice == 0
    } else {
        false
    };
    let codex_global = if has_codex {
        let install_options = vec!["Local (./.codex/)", "Global (~/.codex/)"];
        let install_choice = Select::with_theme(&theme)
            .with_prompt("Codex installation type")
            .items(&install_options)
            .default(0)
            .interact()
            .unwrap();
        install_choice == 1
    } else {
        false
    };
    let generate_commands = if has_claude {
        Confirm::with_theme(&theme)
            .with_prompt("Generate Claude commands for agents?")
            .default(true)
            .interact()
            .unwrap()
    } else {
        false
    };
    let command_indices: Vec<usize> = if has_claude && generate_commands {
        MultiSelect::with_theme(&theme)
            .with_prompt("Select agents to turn into Claude commands (space to toggle, enter to confirm)")
            .items(&selected_names)
            .defaults(&vec![true; selected_names.len()])
            .interact()
            .unwrap()
    } else {
        Vec::new()
    };
    let install_workflow_skill = Confirm::with_theme(&theme)
        .with_prompt("Install workflow skill assets?")
        .default(true)
        .interact()
        .unwrap();

    for platform in selected_platforms {
        match platform {
            Platform::Claude => {
                let agents_target = get_claude_agents_dir(claude_global);
                let commands_target = get_claude_commands_dir(claude_global);
                println!("\nInstalling for Claude...\n");
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
                let mut workflow_installed = false;
                if install_workflow_skill {
                    let skills_target = get_claude_skills_dir(claude_global);
                    match install_workflow_for_claude(&skills_target, &commands_target, &exe_dir) {
                        Ok(_) => {
                            println!("  Installed workflow skill and /ad:wf command");
                            workflow_installed = true;
                        }
                        Err(e) => println!("  Failed to install workflow: {}", e),
                    }
                }
                println!("\nDone for Claude!");
                println!("  Installed {} agents to {:?}", installed_count, agents_target);
                if command_count > 0 {
                    println!("  Created {} commands in {:?}", command_count, commands_target);
                }
                if workflow_installed {
                    let skills_target = get_claude_skills_dir(claude_global);
                    println!("  Installed workflow skill to {:?}", skills_target.join("workflow-skill"));
                    println!("  Installed /ad:wf command to {:?}", commands_target.join("ad").join("wf.md"));
                }
            }
            Platform::Codex => {
                let prompts_target = get_codex_prompts_dir(codex_global);
                println!("\nInstalling for Codex...\n");
                let mut installed_count = 0;
                for agent in &selected_agents {
                    match install_agent(agent, &prompts_target) {
                        Ok(_) => {
                            println!("  Installed prompt: {}", agent.filename);
                            installed_count += 1;
                        }
                        Err(e) => println!("  Failed to install {}: {}", agent.name, e),
                    }
                }
                let mut workflow_installed = false;
                if install_workflow_skill {
                    let skills_target = get_codex_skills_dir(codex_global);
                    match install_workflow_for_codex(&skills_target, &prompts_target, &exe_dir) {
                        Ok(_) => {
                            println!("  Installed workflow skill and workflow prompts");
                            workflow_installed = true;
                        }
                        Err(e) => println!("  Failed to install workflow: {}", e),
                    }
                }
                println!("\nDone for Codex!");
                println!("  Installed {} prompts to {:?}", installed_count, prompts_target);
                if workflow_installed {
                    let skills_target = get_codex_skills_dir(codex_global);
                    println!("  Installed workflow skill to {:?}", skills_target.join("workflow-skill"));
                    println!("  Installed workflow prompt to {:?}", prompts_target.join("workflow.md"));
                    println!("  Installed /ad:wf prompt to {:?}", prompts_target.join("ad").join("wf.md"));
                }
            }
        }
    }
}
