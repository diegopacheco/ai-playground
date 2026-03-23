use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactKind {
    Mcp,
    Hook,
    Command,
    Agent,
    Skill,
    ContextFile,
    MemoryFile,
}

impl ArtifactKind {
    pub fn label(&self) -> &str {
        match self {
            ArtifactKind::Mcp => "MCP",
            ArtifactKind::Hook => "Hook",
            ArtifactKind::Command => "Command",
            ArtifactKind::Agent => "Agent",
            ArtifactKind::Skill => "Skill",
            ArtifactKind::ContextFile => "Context",
            ArtifactKind::MemoryFile => "Memory",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scope {
    Global,
    Project,
}

impl Scope {
    pub fn label(&self) -> &str {
        match self {
            Scope::Global => "global",
            Scope::Project => "project",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Health {
    Active,
    Warning(String),
    Broken(String),
}

impl Health {
    pub fn label(&self) -> &str {
        match self {
            Health::Active => "active",
            Health::Warning(_) => "warning",
            Health::Broken(_) => "broken",
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            Health::Active => "",
            Health::Warning(s) => s.as_str(),
            Health::Broken(s) => s.as_str(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Artifact {
    pub name: String,
    pub kind: ArtifactKind,
    pub scope: Scope,
    pub source_path: PathBuf,
    pub health: Health,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CatalogItem {
    pub name: String,
    pub kind: ArtifactKind,
    pub description: String,
    pub repo_path: PathBuf,
    pub installed: bool,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct BackupEntry {
    pub path: PathBuf,
    pub created_at: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tab {
    Context,
    Mcps,
    Hooks,
    Commands,
    Agents,
    Catalog,
    Backup,
}

impl Tab {
    pub fn all() -> Vec<Tab> {
        vec![
            Tab::Context,
            Tab::Mcps,
            Tab::Hooks,
            Tab::Commands,
            Tab::Agents,
            Tab::Catalog,
            Tab::Backup,
        ]
    }

    pub fn label(&self) -> &str {
        match self {
            Tab::Context => "Context/Memory",
            Tab::Mcps => "MCPs",
            Tab::Hooks => "Hooks",
            Tab::Commands => "Commands",
            Tab::Agents => "Agents/Skills",
            Tab::Catalog => "Catalog",
            Tab::Backup => "Backup/Restore",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Tab::Context => 0,
            Tab::Mcps => 1,
            Tab::Hooks => 2,
            Tab::Commands => 3,
            Tab::Agents => 4,
            Tab::Catalog => 5,
            Tab::Backup => 6,
        }
    }

    pub fn from_index(i: usize) -> Tab {
        match i {
            0 => Tab::Context,
            1 => Tab::Mcps,
            2 => Tab::Hooks,
            3 => Tab::Commands,
            4 => Tab::Agents,
            5 => Tab::Catalog,
            6 => Tab::Backup,
            _ => Tab::Context,
        }
    }
}
