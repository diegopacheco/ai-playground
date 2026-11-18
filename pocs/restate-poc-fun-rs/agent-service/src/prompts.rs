use crate::error::AppError;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub type PromptParams<'a> = BTreeMap<&'a str, String>;

pub fn prompts_dir() -> PathBuf {
    std::env::var("PROMPTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("prompts"))
}

fn split_header(template: &str) -> Option<(&str, &str)> {
    let rest = template.strip_prefix("---\n")?;
    let end = rest.find("\n---\n")?;
    Some((&rest[..end], &rest[end + 5..]))
}

fn declared_params(header: &str) -> Vec<String> {
    header
        .split('$')
        .skip(1)
        .map(|part| part.chars().take_while(|c| c.is_ascii_lowercase() || *c == '_').collect::<String>())
        .filter(|name| !name.is_empty())
        .collect()
}

pub fn render_prompt(template: &str, params: &PromptParams) -> Result<String, AppError> {
    let (header, body) = split_header(template).ok_or_else(|| AppError::new("prompt is missing its --- header"))?;
    let mut declared = declared_params(header);
    let missing: Vec<String> = declared
        .iter()
        .filter(|name| params.get(name.as_str()).is_none_or(|value| value.trim().is_empty()))
        .map(|name| format!("${name}"))
        .collect();
    if !missing.is_empty() {
        return Err(AppError::new(format!("prompt parameters without value: {}", missing.join(", "))));
    }
    declared.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let rendered = declared
        .iter()
        .fold(body.to_string(), |text, name| text.replace(&format!("${name}"), &params[name.as_str()]));
    Ok(rendered.trim().to_string())
}

pub fn load_prompt_from(dir: &Path, name: &str, params: &PromptParams) -> Result<String, AppError> {
    let path = dir.join(format!("{name}.md"));
    let template = std::fs::read_to_string(&path)
        .map_err(|error| AppError::new(format!("cannot read prompt {name}: {error}")))?;
    render_prompt(&template, params)
}

pub fn load_prompt(name: &str, params: &PromptParams) -> Result<String, AppError> {
    load_prompt_from(&prompts_dir(), name, params)
}
