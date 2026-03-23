use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use serde_json::Value;
use crate::model::{Artifact, ArtifactKind, Health, Scope};
use crate::health;

pub fn scan_all() -> Vec<Artifact> {
    let mut artifacts = Vec::new();
    if let Some(home) = dirs::home_dir() {
        let global_claude = home.join(".claude");
        if global_claude.exists() {
            scan_global(&global_claude, &mut artifacts);
        }
    }
    let project_claude = PathBuf::from(".claude");
    if project_claude.exists() {
        scan_project(&project_claude, &mut artifacts);
    }
    let project_claude_md = PathBuf::from("CLAUDE.md");
    if project_claude_md.exists() {
        artifacts.push(Artifact {
            name: "CLAUDE.md".to_string(),
            kind: ArtifactKind::ContextFile,
            scope: Scope::Project,
            source_path: project_claude_md,
            health: Health::Active,
            metadata: HashMap::new(),
        });
    }
    let skills_dir = PathBuf::from("skills");
    if skills_dir.exists() {
        scan_skills_dir(&skills_dir, Scope::Project, &mut artifacts);
    }
    artifacts
}

fn scan_global(claude_dir: &Path, artifacts: &mut Vec<Artifact>) {
    let settings_path = claude_dir.join("settings.json");
    if settings_path.exists() {
        if let Ok(content) = fs::read_to_string(&settings_path) {
            if let Ok(json) = serde_json::from_str::<Value>(&content) {
                scan_mcps_from_json(&json, &settings_path, Scope::Global, artifacts);
                scan_hooks_from_json(&json, &settings_path, Scope::Global, artifacts);
            }
        }
    }
    let claude_md = claude_dir.join("CLAUDE.md");
    if claude_md.exists() {
        artifacts.push(Artifact {
            name: "CLAUDE.md (global)".to_string(),
            kind: ArtifactKind::ContextFile,
            scope: Scope::Global,
            source_path: claude_md,
            health: Health::Active,
            metadata: HashMap::new(),
        });
    }
    let commands_dir = claude_dir.join("commands");
    if commands_dir.exists() {
        scan_file_dir(&commands_dir, ArtifactKind::Command, Scope::Global, artifacts);
    }
    let agents_dir = claude_dir.join("agents");
    if agents_dir.exists() {
        scan_file_dir(&agents_dir, ArtifactKind::Agent, Scope::Global, artifacts);
    }
    let projects_dir = claude_dir.join("projects");
    if projects_dir.exists() {
        scan_memory_dir(&projects_dir, artifacts);
    }
}

fn scan_project(claude_dir: &Path, artifacts: &mut Vec<Artifact>) {
    let settings_path = claude_dir.join("settings.json");
    if settings_path.exists() {
        if let Ok(content) = fs::read_to_string(&settings_path) {
            if let Ok(json) = serde_json::from_str::<Value>(&content) {
                scan_mcps_from_json(&json, &settings_path, Scope::Project, artifacts);
                scan_hooks_from_json(&json, &settings_path, Scope::Project, artifacts);
            }
        }
    }
    let local_settings = claude_dir.join("settings.local.json");
    if local_settings.exists() {
        if let Ok(content) = fs::read_to_string(&local_settings) {
            if let Ok(json) = serde_json::from_str::<Value>(&content) {
                scan_mcps_from_json(&json, &local_settings, Scope::Project, artifacts);
                scan_hooks_from_json(&json, &local_settings, Scope::Project, artifacts);
            }
        }
    }
    let commands_dir = claude_dir.join("commands");
    if commands_dir.exists() {
        scan_file_dir(&commands_dir, ArtifactKind::Command, Scope::Project, artifacts);
    }
    let agents_dir = claude_dir.join("agents");
    if agents_dir.exists() {
        scan_file_dir(&agents_dir, ArtifactKind::Agent, Scope::Project, artifacts);
    }
}

fn scan_mcps_from_json(json: &Value, source: &Path, scope: Scope, artifacts: &mut Vec<Artifact>) {
    if let Some(mcps) = json.get("mcpServers").and_then(|v| v.as_object()) {
        for (name, config) in mcps {
            let mut metadata = HashMap::new();
            if let Some(cmd) = config.get("command").and_then(|v| v.as_str()) {
                metadata.insert("command".to_string(), cmd.to_string());
            }
            if let Some(args) = config.get("args").and_then(|v| v.as_array()) {
                let args_str: Vec<String> = args.iter()
                    .filter_map(|a| a.as_str().map(|s| s.to_string()))
                    .collect();
                metadata.insert("args".to_string(), args_str.join(" "));
            }
            let h = health::check_mcp(&metadata);
            artifacts.push(Artifact {
                name: name.clone(),
                kind: ArtifactKind::Mcp,
                scope: scope.clone(),
                source_path: source.to_path_buf(),
                health: h,
                metadata,
            });
        }
    }
}

fn scan_hooks_from_json(json: &Value, source: &Path, scope: Scope, artifacts: &mut Vec<Artifact>) {
    if let Some(hooks) = json.get("hooks").and_then(|v| v.as_object()) {
        for (event, config) in hooks {
            if let Some(arr) = config.as_array() {
                for (_i, hook) in arr.iter().enumerate() {
                    let mut metadata = HashMap::new();
                    metadata.insert("event".to_string(), event.clone());
                    let cmd_str = hook.get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    metadata.insert("command".to_string(), cmd_str.to_string());
                    let hook_name = if !cmd_str.is_empty() {
                        let short_cmd: String = cmd_str.chars().take(60).collect();
                        format!("{}: {}", event, short_cmd)
                    } else {
                        event.clone()
                    };
                    let h = health::check_hook(&metadata);
                    artifacts.push(Artifact {
                        name: hook_name,
                        kind: ArtifactKind::Hook,
                        scope: scope.clone(),
                        source_path: source.to_path_buf(),
                        health: h,
                        metadata: metadata.clone(),
                    });
                }
            }
        }
    }
}

fn scan_file_dir(dir: &Path, kind: ArtifactKind, scope: Scope, artifacts: &mut Vec<Artifact>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                let mut metadata = HashMap::new();
                if let Ok(content) = fs::read_to_string(&path) {
                    let preview: String = content.chars().take(200).collect();
                    metadata.insert("preview".to_string(), preview);
                }
                artifacts.push(Artifact {
                    name,
                    kind: kind.clone(),
                    scope: scope.clone(),
                    source_path: path,
                    health: Health::Active,
                    metadata,
                });
            } else if path.is_dir() {
                scan_file_dir(&path, kind.clone(), scope.clone(), artifacts);
            }
        }
    }
}

fn scan_skills_dir(dir: &Path, scope: Scope, artifacts: &mut Vec<Artifact>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skill_md = path.join("SKILL.md");
                let name = path.file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                let mut metadata = HashMap::new();
                if skill_md.exists() {
                    if let Ok(content) = fs::read_to_string(&skill_md) {
                        let preview: String = content.chars().take(200).collect();
                        metadata.insert("preview".to_string(), preview);
                    }
                    artifacts.push(Artifact {
                        name,
                        kind: ArtifactKind::Skill,
                        scope: scope.clone(),
                        source_path: path,
                        health: Health::Active,
                        metadata,
                    });
                }
            }
        }
    }
}

fn scan_memory_dir(projects_dir: &Path, artifacts: &mut Vec<Artifact>) {
    for entry in walkdir::WalkDir::new(projects_dir)
        .min_depth(1)
        .into_iter()
        .flatten()
    {
        let path = entry.path();
        if path.is_file() && path.extension().map(|e| e == "md").unwrap_or(false) {
            let name = path.strip_prefix(projects_dir)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            let mut metadata = HashMap::new();
            if let Ok(content) = fs::read_to_string(path) {
                let preview: String = content.chars().take(200).collect();
                metadata.insert("preview".to_string(), preview);
            }
            if let Ok(meta) = fs::metadata(path) {
                metadata.insert("size".to_string(), format!("{}", meta.len()));
            }
            artifacts.push(Artifact {
                name,
                kind: ArtifactKind::MemoryFile,
                scope: Scope::Global,
                source_path: path.to_path_buf(),
                health: Health::Active,
                metadata,
            });
        }
    }
}
