use std::collections::HashMap;
use std::path::Path;
use crate::model::Health;

pub fn check_mcp(metadata: &HashMap<String, String>) -> Health {
    if let Some(cmd) = metadata.get("command") {
        if cmd.is_empty() {
            return Health::Broken("no command specified".to_string());
        }
        let binary = cmd.split_whitespace().next().unwrap_or("");
        if binary.starts_with('/') || binary.starts_with('.') {
            if !Path::new(binary).exists() {
                return Health::Warning(format!("binary not found: {}", binary));
            }
        } else {
            if which_exists(binary) {
                return Health::Active;
            } else {
                return Health::Warning(format!("binary not in PATH: {}", binary));
            }
        }
        Health::Active
    } else {
        Health::Broken("no command in config".to_string())
    }
}

pub fn check_hook(metadata: &HashMap<String, String>) -> Health {
    if let Some(cmd) = metadata.get("command") {
        if cmd.is_empty() {
            return Health::Broken("no command specified".to_string());
        }
        Health::Active
    } else {
        Health::Warning("no command found".to_string())
    }
}

fn which_exists(binary: &str) -> bool {
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(':') {
            let full = Path::new(dir).join(binary);
            if full.exists() {
                return true;
            }
        }
    }
    false
}
