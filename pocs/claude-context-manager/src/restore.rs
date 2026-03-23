use std::fs::{self, File};
use std::path::{Path, PathBuf};
use flate2::read::GzDecoder;
use tar::Archive;

pub fn list_archive_entries(archive_path: &Path) -> Result<Vec<String>, String> {
    let file = File::open(archive_path)
        .map_err(|e| format!("failed to open archive: {}", e))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let mut entries = Vec::new();
    for entry in archive.entries().map_err(|e| format!("failed to read entries: {}", e))? {
        if let Ok(entry) = entry {
            if let Ok(path) = entry.path() {
                entries.push(path.to_string_lossy().to_string());
            }
        }
    }
    Ok(entries)
}

pub fn full_restore(archive_path: &Path) -> Result<String, String> {
    crate::backup::create_backup()
        .map_err(|e| format!("auto-backup before restore failed: {}", e))?;

    let file = File::open(archive_path)
        .map_err(|e| format!("failed to open archive: {}", e))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    let temp_dir = tempfile::tempdir()
        .map_err(|e| format!("failed to create temp dir: {}", e))?;

    archive.unpack(temp_dir.path())
        .map_err(|e| format!("failed to unpack: {}", e))?;

    let global_src = temp_dir.path().join("global");
    if global_src.exists() {
        if let Some(home) = dirs::home_dir() {
            let global_dest = home.join(".claude");
            copy_dir_recursive(&global_src, &global_dest)?;
        }
    }

    let project_claude_src = temp_dir.path().join("project/.claude");
    if project_claude_src.exists() {
        copy_dir_recursive(&project_claude_src, Path::new(".claude"))?;
    }

    let project_md_src = temp_dir.path().join("project/CLAUDE.md");
    if project_md_src.exists() {
        fs::copy(&project_md_src, "CLAUDE.md")
            .map_err(|e| format!("failed to restore CLAUDE.md: {}", e))?;
    }

    let skills_src = temp_dir.path().join("project/skills");
    if skills_src.exists() {
        copy_dir_recursive(&skills_src, Path::new("skills"))?;
    }

    Ok("full restore completed".to_string())
}

pub fn selective_restore(archive_path: &Path, selected: &[String]) -> Result<String, String> {
    crate::backup::create_backup()
        .map_err(|e| format!("auto-backup before restore failed: {}", e))?;

    let file = File::open(archive_path)
        .map_err(|e| format!("failed to open archive: {}", e))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    let temp_dir = tempfile::tempdir()
        .map_err(|e| format!("failed to create temp dir: {}", e))?;

    archive.unpack(temp_dir.path())
        .map_err(|e| format!("failed to unpack: {}", e))?;

    let mut restored = 0;
    for entry_name in selected {
        let src = temp_dir.path().join(entry_name);
        if !src.exists() {
            continue;
        }
        let dest = resolve_restore_path(entry_name)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create dir: {}", e))?;
        }
        fs::copy(&src, &dest)
            .map_err(|e| format!("failed to restore {}: {}", entry_name, e))?;
        restored += 1;
    }

    Ok(format!("restored {} items", restored))
}

fn resolve_restore_path(archive_entry: &str) -> Result<PathBuf, String> {
    if let Some(rest) = archive_entry.strip_prefix("global/") {
        if let Some(home) = dirs::home_dir() {
            return Ok(home.join(".claude").join(rest));
        }
    }
    if let Some(rest) = archive_entry.strip_prefix("project/") {
        return Ok(PathBuf::from(rest));
    }
    Err(format!("unknown archive path: {}", archive_entry))
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
