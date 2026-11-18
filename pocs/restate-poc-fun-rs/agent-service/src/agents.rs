use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

pub trait Agent: Send + Sync {
    fn command(&self, model: &str, prompt: &str) -> Vec<String>;
}

pub struct ClaudeCodeAgent;
pub struct CodexAgent;
pub struct AgyAgent;
pub struct OllamaAgent;

fn owned(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| part.to_string()).collect()
}

impl Agent for ClaudeCodeAgent {
    fn command(&self, model: &str, prompt: &str) -> Vec<String> {
        owned(&["claude", "-p", "--model", model, prompt])
    }
}

impl Agent for CodexAgent {
    fn command(&self, model: &str, prompt: &str) -> Vec<String> {
        owned(&["codex", "exec", "--model", model, prompt])
    }
}

impl Agent for AgyAgent {
    fn command(&self, model: &str, prompt: &str) -> Vec<String> {
        owned(&["agy", "-p", "--model", model, prompt])
    }
}

impl Agent for OllamaAgent {
    fn command(&self, model: &str, prompt: &str) -> Vec<String> {
        owned(&["ollama", "run", model, prompt])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Claude,
    Codex,
    Agy,
    Ollama,
}

impl Provider {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "claude" => Some(Self::Claude),
            "codex" => Some(Self::Codex),
            "agy" => Some(Self::Agy),
            "ollama" => Some(Self::Ollama),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Agy => "agy",
            Self::Ollama => "ollama",
        }
    }

    pub fn agent(&self) -> Box<dyn Agent> {
        match self {
            Self::Claude => Box::new(ClaudeCodeAgent),
            Self::Codex => Box::new(CodexAgent),
            Self::Agy => Box::new(AgyAgent),
            Self::Ollama => Box::new(OllamaAgent),
        }
    }
}

pub trait CommandRunner: Send + Sync {
    fn run(&self, command: Vec<String>) -> impl Future<Output = Result<String, AppError>> + Send;
}

pub async fn call<R: CommandRunner>(agent: &dyn Agent, runner: &R, model: &str, prompt: &str) -> Result<String, AppError> {
    if model.trim().is_empty() {
        return Err(AppError::new("model is required"));
    }
    if prompt.trim().is_empty() {
        return Err(AppError::new("prompt is required"));
    }
    runner.run(agent.command(model, prompt)).await
}

pub struct ProcessRunner;

fn node_binaries(home: &Path) -> Vec<PathBuf> {
    let mut versions: Vec<PathBuf> = std::fs::read_dir(home.join(".nvm").join("versions").join("node"))
        .map(|entries| entries.filter_map(Result::ok).map(|entry| entry.path().join("bin")).collect())
        .unwrap_or_default();
    versions.sort();
    versions.reverse();
    versions
}

pub fn command_path(home: &Path, current: &str) -> String {
    let mut candidates = vec![
        home.join(".local").join("bin"),
        home.join(".bun").join("bin"),
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
    ];
    candidates.extend(node_binaries(home));
    candidates.extend(std::env::split_paths(current));
    let mut unique: Vec<PathBuf> = Vec::new();
    for path in candidates {
        if !path.as_os_str().is_empty() && !unique.contains(&path) {
            unique.push(path);
        }
    }
    std::env::join_paths(unique).map(|joined| joined.to_string_lossy().into_owned()).unwrap_or_default()
}

impl CommandRunner for ProcessRunner {
    async fn run(&self, command: Vec<String>) -> Result<String, AppError> {
        let (program, args) = command.split_first().ok_or_else(|| AppError::new("empty command"))?;
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
        let path = command_path(&home, &std::env::var("PATH").unwrap_or_default());
        let output = Command::new(program)
            .args(args)
            .env("PATH", path)
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|error| AppError::new(format!("Unable to start {program}: {error}")))?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        if output.status.success() {
            return Ok(stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let detail = if stderr.trim().is_empty() { stdout } else { stderr };
        Err(AppError::new(format!("Command failed with {}: {}", output.status, detail.trim())))
    }
}
