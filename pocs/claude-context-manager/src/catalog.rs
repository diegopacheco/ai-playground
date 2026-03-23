use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json::Value;
use crate::model::{ArtifactKind, CatalogItem};

const REPO_URL: &str = "https://github.com/diegopacheco/ai-playground";

pub struct Catalog {
    pub items: Vec<CatalogItem>,
    pub temp_dir: Option<tempfile::TempDir>,
    pub status: CatalogStatus,
}

pub enum CatalogStatus {
    NotLoaded,
    Loading,
    Loaded,
    Error(String),
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            temp_dir: None,
            status: CatalogStatus::NotLoaded,
        }
    }

    pub fn load(&mut self) {
        self.status = CatalogStatus::Loading;
        let temp = match tempfile::tempdir() {
            Ok(t) => t,
            Err(e) => {
                self.status = CatalogStatus::Error(format!("temp dir failed: {}", e));
                return;
            }
        };

        let clone_path = temp.path().join("ai-playground");
        let result = Command::new("git")
            .args(["clone", "--depth", "1", REPO_URL, &clone_path.to_string_lossy()])
            .output();

        match result {
            Ok(output) if output.status.success() => {
                self.items = scan_catalog(&clone_path);
                self.temp_dir = Some(temp);
                self.status = CatalogStatus::Loaded;
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                self.status = CatalogStatus::Error(format!("git clone failed: {}", stderr));
            }
            Err(e) => {
                self.status = CatalogStatus::Error(format!("git not found: {}", e));
            }
        }
    }

    pub fn mark_installed(&mut self, installed_names: &[String]) {
        for item in &mut self.items {
            item.installed = installed_names.contains(&item.name);
        }
    }
}

fn scan_catalog(repo_root: &Path) -> Vec<CatalogItem> {
    let mut items = Vec::new();

    for entry in walkdir::WalkDir::new(repo_root)
        .min_depth(1)
        .max_depth(6)
        .into_iter()
        .flatten()
    {
        let path = entry.path();

        if path.is_dir() && path.join("SKILL.md").exists() {
            let name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let desc = read_skill_description(&path.join("SKILL.md"));
            items.push(CatalogItem {
                name,
                kind: ArtifactKind::Skill,
                description: desc,
                repo_path: path.to_path_buf(),
                installed: false,
                metadata: HashMap::new(),
            });
        }

        if path.is_file() && path.extension().map(|e| e == "md").unwrap_or(false) {
            if let Some(parent) = path.parent() {
                let parent_name = parent.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if parent_name == "commands" {
                    let name = path.file_stem()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    items.push(CatalogItem {
                        name,
                        kind: ArtifactKind::Command,
                        description: read_first_line(path),
                        repo_path: path.to_path_buf(),
                        installed: false,
                        metadata: HashMap::new(),
                    });
                }
                if parent_name == "agents" {
                    let name = path.file_stem()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    items.push(CatalogItem {
                        name,
                        kind: ArtifactKind::Agent,
                        description: read_first_line(path),
                        repo_path: path.to_path_buf(),
                        installed: false,
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        if path.is_file() && path.file_name().map(|n| n == "settings.json").unwrap_or(false) {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(json) = serde_json::from_str::<Value>(&content) {
                    if let Some(mcps) = json.get("mcpServers").and_then(|v| v.as_object()) {
                        for (name, _config) in mcps {
                            items.push(CatalogItem {
                                name: name.clone(),
                                kind: ArtifactKind::Mcp,
                                description: format!("MCP from {}", path.display()),
                                repo_path: path.to_path_buf(),
                                installed: false,
                                metadata: HashMap::new(),
                            });
                        }
                    }
                }
            }
        }
    }

    items.sort_by(|a, b| a.kind.label().cmp(b.kind.label()).then(a.name.cmp(&b.name)));
    items.dedup_by(|a, b| a.name == b.name && a.kind == b.kind);
    items
}

fn read_skill_description(skill_md: &Path) -> String {
    if let Ok(content) = fs::read_to_string(skill_md) {
        for line in content.lines().take(10) {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') && !trimmed.starts_with("---") {
                return trimmed.chars().take(100).collect();
            }
        }
    }
    String::new()
}

fn read_first_line(path: &Path) -> String {
    if let Ok(content) = fs::read_to_string(path) {
        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with("---") {
                return trimmed.chars().take(100).collect();
            }
        }
    }
    String::new()
}

pub fn install_item(item: &CatalogItem, global: bool) -> Result<String, String> {
    match item.kind {
        ArtifactKind::Skill => install_skill(item, global),
        ArtifactKind::Command => install_file(item, "commands", global),
        ArtifactKind::Agent => install_file(item, "agents", global),
        ArtifactKind::Mcp => install_mcp(item, global),
        _ => Err("unsupported artifact type for install".to_string()),
    }
}

fn install_skill(item: &CatalogItem, global: bool) -> Result<String, String> {
    let dest = if global {
        dirs::home_dir()
            .ok_or("no home dir")?
            .join(".claude")
            .join("commands")
            .join(&item.name)
    } else {
        PathBuf::from("skills").join(&item.name)
    };
    copy_dir_recursive(&item.repo_path, &dest)?;
    Ok(format!("installed skill '{}' to {}", item.name, dest.display()))
}

fn install_file(item: &CatalogItem, subdir: &str, global: bool) -> Result<String, String> {
    let dest_dir = if global {
        dirs::home_dir()
            .ok_or("no home dir")?
            .join(".claude")
            .join(subdir)
    } else {
        PathBuf::from(".claude").join(subdir)
    };
    fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("failed to create dir: {}", e))?;
    let filename = item.repo_path.file_name()
        .ok_or("no filename")?;
    let dest = dest_dir.join(filename);
    fs::copy(&item.repo_path, &dest)
        .map_err(|e| format!("failed to copy: {}", e))?;
    Ok(format!("installed '{}' to {}", item.name, dest.display()))
}

fn install_mcp(item: &CatalogItem, global: bool) -> Result<String, String> {
    let source_content = fs::read_to_string(&item.repo_path)
        .map_err(|e| format!("failed to read source: {}", e))?;
    let source_json: Value = serde_json::from_str(&source_content)
        .map_err(|e| format!("failed to parse source JSON: {}", e))?;
    let mcp_config = source_json
        .get("mcpServers")
        .and_then(|m| m.get(&item.name))
        .ok_or("MCP config not found in source")?
        .clone();

    let settings_path = if global {
        dirs::home_dir()
            .ok_or("no home dir")?
            .join(".claude")
            .join("settings.json")
    } else {
        PathBuf::from(".claude").join("settings.json")
    };

    let mut json: Value = if settings_path.exists() {
        let content = fs::read_to_string(&settings_path)
            .map_err(|e| format!("failed to read settings: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse settings: {}", e))?
    } else {
        if let Some(parent) = settings_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create dir: {}", e))?;
        }
        serde_json::json!({})
    };

    if json.get("mcpServers").is_none() {
        json["mcpServers"] = serde_json::json!({});
    }
    json["mcpServers"][&item.name] = mcp_config;

    let output = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialize: {}", e))?;
    fs::write(&settings_path, output)
        .map_err(|e| format!("failed to write: {}", e))?;

    Ok(format!("installed MCP '{}' to {}", item.name, settings_path.display()))
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest)
        .map_err(|e| format!("failed to create {}: {}", dest.display(), e))?;
    for entry in walkdir::WalkDir::new(src).into_iter().flatten() {
        let path = entry.path();
        let rel = path.strip_prefix(src)
            .map_err(|e| format!("strip prefix: {}", e))?;
        let target = dest.join(rel);
        if path.is_dir() {
            fs::create_dir_all(&target)
                .map_err(|e| format!("mkdir {}: {}", target.display(), e))?;
        } else {
            fs::copy(path, &target)
                .map_err(|e| format!("copy {}: {}", path.display(), e))?;
        }
    }
    Ok(())
}
