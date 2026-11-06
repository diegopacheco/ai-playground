use crate::error::Result;
use crate::frontmatter;
use crate::kind::Kind;
use crate::picker::Entry;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const SKIPPED_DIRS: [&str; 4] = [".git", "node_modules", "target", ".run"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub kind: Kind,
    pub name: String,
    pub description: String,
    pub path: String,
    pub source: PathBuf,
    pub marketplace: Option<String>,
}

impl Item {
    pub fn entry(&self) -> Entry {
        Entry {
            name: self.name.clone(),
            haystack: format!("{} {} {} {}", self.kind.label(), self.name, self.path, self.description).to_lowercase(),
        }
    }

    pub fn matches(&self, selector: &str) -> bool {
        match selector.split_once(':') {
            Some((kind, name)) => Kind::parse(kind) == Some(self.kind) && self.name == name,
            None => self.name == selector,
        }
    }
}

pub fn scan(root: &Path) -> Result<Vec<Item>> {
    let mut items = Vec::new();
    walk(root, root, &mut items)?;
    items.extend(plugins(root)?);
    let mut seen = HashSet::new();
    items.retain(|item| seen.insert((item.kind, item.name.clone())));
    items.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.name.cmp(&b.name)));
    Ok(items)
}

fn walk(root: &Path, dir: &Path, items: &mut Vec<Item>) -> Result<()> {
    let dir_name = file_name(dir);
    let skill = dir.join("SKILL.md");
    if skill.is_file() {
        let meta = frontmatter::parse(&fs::read_to_string(&skill)?);
        let name = meta.name.unwrap_or_else(|| dir_name.clone());
        let command = dir.join("command.md");
        if command.is_file() {
            items.push(markdown_item(root, Kind::Command, &command, Some(name.clone()))?);
        }
        items.push(Item {
            kind: Kind::Skill,
            name,
            description: meta.description.unwrap_or_default(),
            path: relative(root, dir),
            source: dir.to_path_buf(),
            marketplace: None,
        });
    }
    let folder_kind = match dir_name.as_str() {
        "commands" => Some(Kind::Command),
        "agents" => Some(Kind::Agent),
        _ => None,
    };
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?.map(|entry| entry.map(|e| e.path())).collect::<std::io::Result<_>>()?;
    entries.sort();
    for path in entries {
        if path.is_dir() {
            if !SKIPPED_DIRS.contains(&file_name(&path).as_str()) {
                walk(root, &path, items)?;
            }
        } else if let Some(kind) = folder_kind
            && is_markdown(&path)
            && file_name(&path) != "README.md"
        {
            items.push(markdown_item(root, kind, &path, None)?);
        }
    }
    Ok(())
}

fn markdown_item(root: &Path, kind: Kind, path: &Path, name: Option<String>) -> Result<Item> {
    let text = fs::read_to_string(path)?;
    let meta = frontmatter::parse(&text);
    let slash = frontmatter::slash_name(&text).filter(|_| kind == Kind::Command);
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let name = slash.or(name).or(meta.name.filter(|_| kind == Kind::Agent)).unwrap_or(stem);
    Ok(Item {
        kind,
        name,
        description: meta.description.unwrap_or_default(),
        path: relative(root, path),
        source: path.to_path_buf(),
        marketplace: None,
    })
}

fn plugins(root: &Path) -> Result<Vec<Item>> {
    let manifest = root.join(".claude-plugin").join("marketplace.json");
    if !manifest.is_file() {
        return Ok(Vec::new());
    }
    let json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&manifest)?)?;
    let marketplace = json["name"].as_str().map(str::to_string);
    let list = json["plugins"].as_array().cloned().unwrap_or_default();
    Ok(list
        .iter()
        .filter_map(|plugin| {
            let name = plugin["name"].as_str()?.to_string();
            Some(Item {
                kind: Kind::Plugin,
                path: plugin["source"].as_str().unwrap_or(&name).trim_start_matches("./").to_string(),
                name,
                description: plugin["description"].as_str().unwrap_or_default().to_string(),
                source: root.to_path_buf(),
                marketplace: marketplace.clone(),
            })
        })
        .collect())
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map(|rel| rel.to_string_lossy().to_string()).unwrap_or_default()
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

fn is_markdown(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "md")
}
