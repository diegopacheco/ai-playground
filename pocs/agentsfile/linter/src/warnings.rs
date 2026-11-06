use std::path::Path;

use agentsfile::{Attribute, Block, Statement, Value};

use crate::diagnostic::Diagnostic;

const PROFILE_KINDS: [&str; 5] = ["skill", "agent", "command", "mcp", "plugin"];
const PLUGIN_KINDS: [&str; 4] = ["skill", "agent", "command", "mcp"];
const SECRET_KEY_HINTS: [&str; 5] = ["TOKEN", "SECRET", "PASSWORD", "KEY", "AUTHORIZATION"];
const SECRET_VALUE_PREFIXES: [&str; 5] = ["sk-", "ghp_", "github_pat_", "xox", "AKIA"];

pub fn check(statements: &[Statement], base: &Path) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for profile in blocks(statements).filter(|b| b.kind == "profile") {
        check_name(profile, &mut out);
        let items: Vec<&Block> = blocks(&profile.body).filter(|b| PROFILE_KINDS.contains(&b.kind.as_str())).collect();
        if items.is_empty() {
            out.push(Diagnostic::warning(
                "W001",
                profile.line,
                profile.col,
                format!("profile \"{}\" declares nothing", profile.label),
            ));
        }
        for item in items {
            check_item(item, base, &mut out);
        }
    }
    out
}

fn check_item(item: &Block, base: &Path, out: &mut Vec<Diagnostic>) {
    check_name(item, out);
    check_source(item, base, out);
    check_content(item, out);
    match item.kind.as_str() {
        "mcp" => {
            check_secrets(item, out);
            check_ignored(item, out);
        }
        "skill" => {
            for file in blocks(&item.body).filter(|b| b.kind == "file") {
                check_source(file, base, out);
                check_content(file, out);
            }
        }
        "plugin" => {
            for child in blocks(&item.body).filter(|b| PLUGIN_KINDS.contains(&b.kind.as_str())) {
                check_item(child, base, out);
            }
        }
        _ => {}
    }
}

fn blocks(statements: &[Statement]) -> impl Iterator<Item = &Block> {
    statements.iter().filter_map(|s| match s {
        Statement::Block(b) => Some(b),
        Statement::Attribute(_) => None,
    })
}

fn attributes(block: &Block) -> impl Iterator<Item = &Attribute> {
    block.body.iter().filter_map(|s| match s {
        Statement::Attribute(a) => Some(a),
        Statement::Block(_) => None,
    })
}

fn string_attr<'a>(block: &'a Block, name: &str) -> Option<(&'a Attribute, &'a str)> {
    attributes(block).find(|a| a.name == name).and_then(|a| match &a.value {
        Value::Str(s) => Some((a, s.as_str())),
        _ => None,
    })
}

pub fn is_kebab_case(name: &str) -> bool {
    !name.is_empty()
        && name.split('-').all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()))
}

pub fn is_local_path(source: &str) -> bool {
    source.starts_with('.') || source.starts_with('/')
}

fn check_name(block: &Block, out: &mut Vec<Diagnostic>) {
    if !is_kebab_case(&block.label) {
        out.push(Diagnostic::warning(
            "W004",
            block.line,
            block.col,
            format!("{} name \"{}\" is not kebab-case", block.kind, block.label),
        ));
    }
}

fn check_source(block: &Block, base: &Path, out: &mut Vec<Diagnostic>) {
    let Some((attr, source)) = string_attr(block, "source") else { return };
    if !is_local_path(source) || base.join(source).exists() {
        return;
    }
    out.push(Diagnostic::warning(
        "W002",
        attr.value_line,
        attr.value_col,
        format!("source \"{source}\" does not exist"),
    ));
}

fn check_content(block: &Block, out: &mut Vec<Diagnostic>) {
    let Some((attr, content)) = string_attr(block, "content") else { return };
    if content.trim().is_empty() {
        out.push(Diagnostic::warning(
            "W006",
            attr.value_line,
            attr.value_col,
            format!("content of {} \"{}\" is empty", block.kind, block.label),
        ));
    }
}

pub fn looks_like_secret(key: &str, value: &str) -> bool {
    if value.contains("${") || value.is_empty() {
        return false;
    }
    let upper = key.to_ascii_uppercase();
    SECRET_KEY_HINTS.iter().any(|h| upper.contains(h)) || SECRET_VALUE_PREFIXES.iter().any(|p| value.starts_with(p))
}

fn check_secrets(block: &Block, out: &mut Vec<Diagnostic>) {
    for attr in attributes(block).filter(|a| a.name == "env" || a.name == "headers") {
        let Value::Map(entries) = &attr.value else { continue };
        for (key, _) in entries.iter().filter(|(k, v)| looks_like_secret(k, v)) {
            out.push(Diagnostic::warning(
                "W003",
                attr.value_line,
                attr.value_col,
                format!("{} \"{key}\" looks like a hardcoded secret, use ${{NAME}}", attr.name),
            ));
        }
    }
}

fn check_ignored(block: &Block, out: &mut Vec<Diagnostic>) {
    let Some((_, transport)) = string_attr(block, "transport") else { return };
    let ignored: &[&str] = match transport {
        "stdio" => &["url", "headers"],
        "http" | "sse" => &["command", "args", "env"],
        _ => return,
    };
    for attr in attributes(block).filter(|a| ignored.contains(&a.name.as_str())) {
        out.push(Diagnostic::warning(
            "W005",
            attr.line,
            attr.col,
            format!("attribute \"{}\" is ignored with {transport} transport", attr.name),
        ));
    }
}
