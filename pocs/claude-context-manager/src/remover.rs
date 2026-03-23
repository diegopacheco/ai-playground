use std::fs;
use std::path::Path;
use serde_json::Value;
use crate::model::{Artifact, ArtifactKind};

pub fn remove_artifact(artifact: &Artifact) -> Result<String, String> {
    match artifact.kind {
        ArtifactKind::Mcp => remove_json_key(&artifact.source_path, "mcpServers", &artifact.name),
        ArtifactKind::Hook => remove_hook(artifact),
        ArtifactKind::Command | ArtifactKind::Agent => remove_file(&artifact.source_path),
        ArtifactKind::Skill => remove_directory(&artifact.source_path),
        ArtifactKind::ContextFile | ArtifactKind::MemoryFile => remove_file(&artifact.source_path),
    }
}

fn remove_json_key(settings_path: &Path, section: &str, key: &str) -> Result<String, String> {
    let content = fs::read_to_string(settings_path)
        .map_err(|e| format!("failed to read {}: {}", settings_path.display(), e))?;
    let mut json: Value = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse JSON: {}", e))?;
    if let Some(obj) = json.get_mut(section).and_then(|v| v.as_object_mut()) {
        obj.remove(key);
    } else {
        return Err(format!("section '{}' not found", section));
    }
    let output = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialize JSON: {}", e))?;
    fs::write(settings_path, output)
        .map_err(|e| format!("failed to write {}: {}", settings_path.display(), e))?;
    Ok(format!("removed '{}' from {}", key, settings_path.display()))
}

fn remove_hook(artifact: &Artifact) -> Result<String, String> {
    let event = artifact.metadata.get("event")
        .ok_or("no event in metadata")?;
    let content = fs::read_to_string(&artifact.source_path)
        .map_err(|e| format!("failed to read: {}", e))?;
    let mut json: Value = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse JSON: {}", e))?;
    if let Some(hooks) = json.get_mut("hooks").and_then(|v| v.as_object_mut()) {
        if let Some(arr) = hooks.get_mut(event).and_then(|v| v.as_array_mut()) {
            if let Some(cmd) = artifact.metadata.get("command") {
                arr.retain(|h| {
                    h.get("command").and_then(|v| v.as_str()) != Some(cmd.as_str())
                });
            }
            if arr.is_empty() {
                hooks.remove(event);
            }
        }
    }
    let output = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialize: {}", e))?;
    fs::write(&artifact.source_path, output)
        .map_err(|e| format!("failed to write: {}", e))?;
    Ok(format!("removed hook '{}' from {}", artifact.name, artifact.source_path.display()))
}

fn remove_file(path: &Path) -> Result<String, String> {
    fs::remove_file(path)
        .map_err(|e| format!("failed to remove {}: {}", path.display(), e))?;
    Ok(format!("removed {}", path.display()))
}

fn remove_directory(path: &Path) -> Result<String, String> {
    fs::remove_dir_all(path)
        .map_err(|e| format!("failed to remove {}: {}", path.display(), e))?;
    Ok(format!("removed {}", path.display()))
}
