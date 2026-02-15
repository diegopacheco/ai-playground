use crate::state::State;
use regex::Regex;

pub fn apply_transform(operation: &str, input: &str, state: &State) -> Result<String, String> {
    match operation {
        "json_extract" => json_extract(input, state),
        "regex" => regex_extract(input, state),
        "template" => Ok(state.resolve_template(input)),
        "uppercase" => Ok(input.to_uppercase()),
        "lowercase" => Ok(input.to_lowercase()),
        "trim" => Ok(input.trim().to_string()),
        _ => Err(format!("unknown transform operation: {}", operation)),
    }
}

fn json_extract(input: &str, state: &State) -> Result<String, String> {
    let path = state.get("json_path").unwrap_or("$");
    let json: serde_json::Value =
        serde_json::from_str(input).map_err(|e| format!("invalid JSON: {}", e))?;
    let keys: Vec<&str> = path
        .trim_start_matches('$')
        .trim_start_matches('.')
        .split('.')
        .filter(|s| !s.is_empty())
        .collect();
    let mut current = &json;
    for key in keys {
        current = current
            .get(key)
            .ok_or_else(|| format!("key '{}' not found", key))?;
    }
    match current {
        serde_json::Value::String(s) => Ok(s.clone()),
        _ => Ok(current.to_string()),
    }
}

fn regex_extract(input: &str, state: &State) -> Result<String, String> {
    let pattern = state
        .get("regex_pattern")
        .ok_or_else(|| "regex_pattern not set in state".to_string())?;
    let re = Regex::new(pattern).map_err(|e| format!("invalid regex: {}", e))?;
    if let Some(caps) = re.captures(input) {
        if let Some(m) = caps.get(1) {
            Ok(m.as_str().to_string())
        } else if let Some(m) = caps.get(0) {
            Ok(m.as_str().to_string())
        } else {
            Ok(String::new())
        }
    } else {
        Ok(String::new())
    }
}
