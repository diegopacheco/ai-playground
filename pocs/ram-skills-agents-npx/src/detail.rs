use crate::catalog::Item;
use crate::error::Result;
use crate::kind::{Kind, Target};
use crate::report::tilde;
use std::fs;
use std::path::Path;

const MAX_FILES: usize = 40;
const SKIPPED: [&str; 2] = [".git", "node_modules"];

#[derive(Debug, PartialEq, Eq)]
pub struct Detail {
    pub facts: Vec<(String, String)>,
    pub files: Vec<String>,
    pub body: Vec<String>,
}

pub fn load(item: &Item, targets: &[Target], home: &Path) -> Result<Detail> {
    let mut facts = vec![("path".to_string(), item.path.clone())];
    for target in targets {
        facts.push((format!("→ {}", target.label()), destination(item, *target, home)));
    }
    let (files, body) = match item.kind {
        Kind::Skill => (list_files(&item.source)?, read(&item.source.join("SKILL.md"))?),
        Kind::Command | Kind::Agent => (Vec::new(), read(&item.source)?),
        Kind::Plugin => plugin_body(item)?,
    };
    Ok(Detail { facts, files, body: body.lines().map(str::to_string).collect() })
}

pub fn destination(item: &Item, target: Target, home: &Path) -> String {
    if !target.supports(item.kind) {
        return format!("not supported by {}, skipped", target.label());
    }
    let dir = target.dir(home, item.kind);
    let path = match item.kind {
        Kind::Skill => dir.join(&item.name),
        Kind::Command | Kind::Agent => dir.join(format!("{}.md", item.name)),
        Kind::Plugin => {
            let marketplace = item.marketplace.as_deref().unwrap_or("marketplace");
            return format!("claude plugin install {}@{}", item.name, marketplace);
        }
    };
    tilde(&path.to_string_lossy(), home)
}

fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()).into())
}

fn plugin_body(item: &Item) -> Result<(Vec<String>, String)> {
    let dir = item.source.join(&item.path);
    let readme = dir.join("README.md");
    let files = if dir.is_dir() { list_files(&dir)? } else { Vec::new() };
    let body = if readme.is_file() { read(&readme)? } else { item.description.clone() };
    Ok((files, body))
}

fn list_files(root: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if SKIPPED.contains(&name.as_str()) {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(rel) = path.strip_prefix(root) {
                files.push(rel.to_string_lossy().to_string());
            }
        }
    }
    files.sort();
    let total = files.len();
    if total > MAX_FILES {
        files.truncate(MAX_FILES);
        files.push(format!("… and {} more", total - MAX_FILES));
    }
    Ok(files)
}
