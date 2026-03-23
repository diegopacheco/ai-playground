use std::fs::{self, File};
use std::path::{Path, PathBuf};
use chrono::Local;
use flate2::Compression;
use flate2::write::GzEncoder;
use tar::Builder;
use crate::model::BackupEntry;

pub fn backup_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".claude-backups"))
        .unwrap_or_else(|| PathBuf::from(".claude-backups"))
}

pub fn create_backup() -> Result<String, String> {
    let dest_dir = backup_dir();
    fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("failed to create backup dir: {}", e))?;

    let timestamp = Local::now().format("%Y-%m-%d-%H%M%S");
    let filename = format!("backup-{}.tar.gz", timestamp);
    let dest_path = dest_dir.join(&filename);

    let file = File::create(&dest_path)
        .map_err(|e| format!("failed to create archive: {}", e))?;
    let enc = GzEncoder::new(file, Compression::default());
    let mut builder = Builder::new(enc);

    if let Some(home) = dirs::home_dir() {
        let global_claude = home.join(".claude");
        if global_claude.exists() {
            add_dir_to_archive(&mut builder, &global_claude, Path::new("global"))?;
        }
    }

    let project_claude = PathBuf::from(".claude");
    if project_claude.exists() {
        add_dir_to_archive(&mut builder, &project_claude, Path::new("project/.claude"))?;
    }

    let project_claude_md = PathBuf::from("CLAUDE.md");
    if project_claude_md.exists() {
        builder.append_path_with_name(&project_claude_md, "project/CLAUDE.md")
            .map_err(|e| format!("failed to add CLAUDE.md: {}", e))?;
    }

    let skills_dir = PathBuf::from("skills");
    if skills_dir.exists() {
        add_dir_to_archive(&mut builder, &skills_dir, Path::new("project/skills"))?;
    }

    builder.finish()
        .map_err(|e| format!("failed to finish archive: {}", e))?;

    Ok(format!("backup created: {}", dest_path.display()))
}

fn add_dir_to_archive<W: std::io::Write>(
    builder: &mut Builder<W>,
    src_dir: &Path,
    archive_prefix: &Path,
) -> Result<(), String> {
    for entry in walkdir::WalkDir::new(src_dir)
        .into_iter()
        .flatten()
    {
        let path = entry.path();
        if path.is_file() {
            let rel = path.strip_prefix(src_dir)
                .map_err(|e| format!("strip prefix error: {}", e))?;
            let archive_path = archive_prefix.join(rel);
            builder.append_path_with_name(path, &archive_path)
                .map_err(|e| format!("failed to add {}: {}", path.display(), e))?;
        }
    }
    Ok(())
}

pub fn list_backups() -> Vec<BackupEntry> {
    let dir = backup_dir();
    if !dir.exists() {
        return Vec::new();
    }
    let mut entries = Vec::new();
    if let Ok(read) = fs::read_dir(&dir) {
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "gz").unwrap_or(false) {
                let name = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let created_at = name
                    .strip_prefix("backup-")
                    .and_then(|s| s.strip_suffix(".tar.gz"))
                    .unwrap_or(&name)
                    .to_string();
                let size_bytes = fs::metadata(&path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                entries.push(BackupEntry { path, created_at, size_bytes });
            }
        }
    }
    entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    entries
}
