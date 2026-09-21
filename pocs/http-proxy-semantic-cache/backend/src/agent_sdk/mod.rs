pub mod json;

use json::Json;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const UNAVAILABLE: &str = "unavailable";
const NEVER: &str = "never";
const USAGE_URL: &str = "https://api.anthropic.com/api/oauth/usage";
const SESSION_WINDOW_MINUTES: f64 = 360.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelQuota {
    pub label: String,
    pub remaining: String,
    pub resets_in: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub weekly_limit: String,
    pub current_limit: String,
    pub time_to_reset_week: String,
    pub time_to_reset_session: String,
    pub models: Vec<ModelQuota>,
}

impl Default for Usage {
    fn default() -> Self {
        Self {
            weekly_limit: UNAVAILABLE.into(),
            current_limit: UNAVAILABLE.into(),
            time_to_reset_week: UNAVAILABLE.into(),
            time_to_reset_session: UNAVAILABLE.into(),
            models: Vec::new(),
        }
    }
}

pub type Runner = fn(Vec<String>) -> Result<String, String>;

pub trait Agent {
    fn call(&self, model: &str, prompt: &str, args: &[String]) -> Result<String, String>;
    fn usage(&self) -> Usage {
        Usage::default()
    }
}

pub fn command_path(command: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut paths: Vec<String> = Vec::new();
    if command.contains('/') {
        if let Some(parent) = Path::new(command).parent() {
            paths.push(parent.to_string_lossy().into_owned());
        }
    }
    paths.push(format!("{home}/.local/bin"));
    paths.push(format!("{home}/.bun/bin"));
    paths.push("/opt/homebrew/bin".into());
    paths.push("/usr/local/bin".into());
    if let Ok(entries) = fs::read_dir(format!("{home}/.nvm/versions/node")) {
        let mut versions: Vec<String> = entries.filter_map(|entry| entry.ok().map(|value| value.path().join("bin").to_string_lossy().into_owned())).collect();
        versions.sort();
        versions.reverse();
        paths.extend(versions);
    }
    paths.extend(std::env::var("PATH").unwrap_or_default().split(':').map(String::from));
    let mut unique: Vec<String> = Vec::new();
    for path in paths {
        if !path.is_empty() && !unique.contains(&path) {
            unique.push(path);
        }
    }
    unique.join(":")
}

pub fn execute(command: Vec<String>) -> Result<String, String> {
    let output = Command::new(&command[0])
        .args(&command[1..])
        .stdin(Stdio::null())
        .env("PATH", command_path(&command[0]))
        .output()
        .map_err(|error| format!("Unable to start {}: {error}", command[0]))?;
    if !output.status.success() {
        let errors = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("Command failed with {}: {errors}", output.status));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn validate(model: &str, prompt: &str) -> Result<(), String> {
    if model.trim().is_empty() {
        return Err("model is required".into());
    }
    if prompt.trim().is_empty() {
        return Err("prompt is required".into());
    }
    Ok(())
}

pub fn format_duration(millis: i64) -> String {
    if millis <= 0 {
        return "0m".into();
    }
    let total_minutes = millis / 60000;
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;
    if hours > 0 { format!("{hours}h {minutes}m") } else { format!("{minutes}m") }
}

pub fn format_percent(value: f64) -> String {
    format!("{}%", value.clamp(0.0, 100.0).round() as i64)
}

fn now_millis() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|value| value.as_millis() as i64).unwrap_or(0)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

pub fn parse_iso_millis(text: &str) -> Option<i64> {
    let characters: Vec<char> = text.chars().collect();
    if characters.len() < 19 {
        return None;
    }
    let number = |start: usize, length: usize| -> Option<i64> { text.get(start..start + length)?.parse::<i64>().ok() };
    let year = number(0, 4)?;
    let month = number(5, 2)?;
    let day = number(8, 2)?;
    let hour = number(11, 2)?;
    let minute = number(14, 2)?;
    let second = number(17, 2)?;
    let mut millis = (days_from_civil(year, month, day) * 86400 + hour * 3600 + minute * 60 + second) * 1000;
    let offset = text.rfind(['+', '-']).filter(|position| *position > 10);
    if let Some(position) = offset {
        let sign = if text[position..].starts_with('-') { -1 } else { 1 };
        let hours = text.get(position + 1..position + 3)?.parse::<i64>().ok()?;
        let minutes = text.get(position + 4..position + 6).and_then(|value| value.parse::<i64>().ok()).unwrap_or(0);
        millis -= sign * (hours * 3600 + minutes * 60) * 1000;
    }
    Some(millis)
}

fn duration_until(moment: Option<&Json>) -> String {
    match moment {
        Some(Json::Number(seconds)) => format_duration((*seconds as i64) * 1000 - now_millis()),
        Some(Json::Text(text)) => match parse_iso_millis(text) {
            Some(millis) => format_duration(millis - now_millis()),
            None => UNAVAILABLE.into(),
        },
        _ => UNAVAILABLE.into(),
    }
}

fn window_usage(windows: Vec<(&str, &Json)>, used_key: &str, reset_key: &str) -> Usage {
    let mut usage = Usage::default();
    for (prefix, window) in windows {
        let used = match window.get(used_key).and_then(Json::number) {
            Some(value) => value,
            None => continue,
        };
        let limit = format_percent(100.0 - used);
        let reset = duration_until(window.get(reset_key));
        if prefix == "session" {
            usage.current_limit = limit;
            usage.time_to_reset_session = reset;
        } else {
            usage.weekly_limit = limit;
            usage.time_to_reset_week = reset;
        }
    }
    usage
}

macro_rules! agent {
    ($name:ident, $builder:expr) => {
        pub struct $name;
        impl $name {
            pub fn command(model: &str, prompt: &str, args: &[String]) -> Vec<String> {
                $builder(model, prompt, args)
            }
        }
    };
}

agent!(ClaudeCodeAgent, |model: &str, prompt: &str, args: &[String]| [vec!["claude".into(), "-p".into(), "--model".into(), model.into()], args.to_vec(), vec![prompt.into()]].concat());
agent!(CodexAgent, |model: &str, prompt: &str, args: &[String]| [vec!["codex".into(), "exec".into(), "--model".into(), model.into()], args.to_vec(), vec![prompt.into()]].concat());
agent!(AgyAgent, |model: &str, prompt: &str, args: &[String]| [vec!["agy".into(), "-p".into(), "--model".into(), model.into()], args.to_vec(), vec![prompt.into()]].concat());
agent!(OllamaAgent, |model: &str, prompt: &str, args: &[String]| [vec!["ollama".into(), "run".into()], args.to_vec(), vec![model.into(), prompt.into()]].concat());

pub fn curl_config(token: &str) -> String {
    let headers = [format!("Authorization: Bearer {token}"), "anthropic-beta: oauth-2025-04-20".into(), "Accept: application/json".into()];
    let mut config = format!("url = \"{USAGE_URL}\"\n");
    for header in headers {
        config.push_str(&format!("header = \"{header}\"\n"));
    }
    config
}

pub fn access_token(credentials: &Json) -> String {
    credentials.get("claudeAiOauth").and_then(|oauth| oauth.get("accessToken")).and_then(Json::text).unwrap_or_default().to_string()
}

pub fn parse_claude_usage(data: &Json) -> Usage {
    let mut windows = Vec::new();
    if let Some(window) = data.get("five_hour") {
        windows.push(("session", window));
    }
    if let Some(window) = data.get("seven_day") {
        windows.push(("week", window));
    }
    window_usage(windows, "utilization", "resets_at")
}

impl ClaudeCodeAgent {
    fn token(&self, runner: Runner) -> String {
        if let Ok(configured) = std::env::var("CLAUDE_CODE_OAUTH_TOKEN") {
            if !configured.is_empty() {
                return configured;
            }
        }
        if let Ok(output) = runner(vec!["security".into(), "find-generic-password".into(), "-s".into(), "Claude Code-credentials".into(), "-w".into()]) {
            let token = access_token(&Json::object(&output));
            if !token.is_empty() {
                return token;
            }
        }
        let home = std::env::var("CLAUDE_CONFIG_DIR").unwrap_or_else(|_| format!("{}/.claude", std::env::var("HOME").unwrap_or_default()));
        match fs::read_to_string(Path::new(&home).join(".credentials.json")) {
            Ok(content) => access_token(&Json::object(&content)),
            Err(_) => String::new(),
        }
    }

    fn read_usage(&self, runner: Runner, token: &str) -> Result<String, String> {
        let directory = std::env::temp_dir().join(format!("agent-sdk-{}-{}", std::process::id(), now_millis()));
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let path = directory.join("usage.curl");
        let result = (|| {
            fs::write(&path, curl_config(token)).map_err(|error| error.to_string())?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).map_err(|error| error.to_string())?;
            runner(vec!["curl".into(), "-sS".into(), "-K".into(), path.to_string_lossy().into_owned()])
        })();
        let _ = fs::remove_dir_all(&directory);
        result
    }

    pub fn usage_with(&self, runner: Runner) -> Usage {
        let token = self.token(runner);
        if token.is_empty() {
            return Usage::default();
        }
        match self.read_usage(runner, &token) {
            Ok(output) => parse_claude_usage(&Json::object(&output)),
            Err(_) => Usage::default(),
        }
    }
}

impl Agent for ClaudeCodeAgent {
    fn call(&self, model: &str, prompt: &str, args: &[String]) -> Result<String, String> {
        validate(model, prompt)?;
        execute(Self::command(model, prompt, args))
    }

    fn usage(&self) -> Usage {
        self.usage_with(execute)
    }
}

fn session_files(root: &Path) -> Vec<PathBuf> {
    let mut found: Vec<(PathBuf, SystemTime)> = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().map(|value| value == "jsonl").unwrap_or(false) {
                let modified = entry.metadata().and_then(|value| value.modified()).unwrap_or(UNIX_EPOCH);
                found.push((path, modified));
            }
        }
    }
    found.sort_by(|left, right| right.1.cmp(&left.1));
    found.into_iter().take(20).map(|(path, _)| path).collect()
}

pub fn latest_rate_limits(home: &Path) -> Json {
    for path in session_files(&home.join("sessions")) {
        let content = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(_) => continue,
        };
        for line in content.lines().rev() {
            let limits = Json::object(line).get("payload").and_then(|payload| payload.get("rate_limits")).cloned();
            if let Some(limits) = limits {
                let has_window = matches!(limits.get("primary"), Some(Json::Object(_))) || matches!(limits.get("secondary"), Some(Json::Object(_)));
                if has_window {
                    return limits;
                }
            }
        }
    }
    Json::Object(Vec::new())
}

pub fn parse_codex_usage(limits: &Json) -> Usage {
    let mut windows = Vec::new();
    for key in ["primary", "secondary"] {
        if let Some(window @ Json::Object(_)) = limits.get(key) {
            let minutes = window.get("window_minutes").and_then(Json::number).unwrap_or(f64::MAX);
            windows.push((if minutes <= SESSION_WINDOW_MINUTES { "session" } else { "week" }, window));
        }
    }
    window_usage(windows, "used_percent", "resets_at")
}

impl Agent for CodexAgent {
    fn call(&self, model: &str, prompt: &str, args: &[String]) -> Result<String, String> {
        validate(model, prompt)?;
        execute(Self::command(model, prompt, args))
    }

    fn usage(&self) -> Usage {
        let home = std::env::var("CODEX_HOME").unwrap_or_else(|_| format!("{}/.codex", std::env::var("HOME").unwrap_or_default()));
        parse_codex_usage(&latest_rate_limits(Path::new(&home)))
    }
}

pub fn parse_agy_usage(data: &Json) -> Usage {
    let models = match data.get("models").and_then(Json::list) {
        Some(models) => models,
        None => return Usage::default(),
    };
    let mut order: Vec<String> = Vec::new();
    let mut quotas: Vec<(String, f64, i64)> = Vec::new();
    for model in models {
        if model.get("isAutocompleteOnly").map(Json::is_true).unwrap_or(false) {
            continue;
        }
        let remaining = match model.get("remainingPercentage").and_then(Json::number) {
            Some(value) => value,
            None => continue,
        };
        let label = model
            .get("label")
            .and_then(Json::text)
            .filter(|value| !value.is_empty())
            .or_else(|| model.get("modelId").and_then(Json::text))
            .unwrap_or_default()
            .to_string();
        if label.is_empty() {
            continue;
        }
        let reset = model.get("timeUntilResetMs").and_then(Json::number).unwrap_or(0.0) as i64;
        match quotas.iter_mut().find(|(name, _, _)| *name == label) {
            Some(entry) if remaining < entry.1 => {
                entry.1 = remaining;
                entry.2 = reset;
            }
            Some(_) => {}
            None => {
                order.push(label.clone());
                quotas.push((label, remaining, reset));
            }
        }
    }
    if quotas.is_empty() {
        return Usage::default();
    }
    let mut usage = Usage::default();
    let mut lowest = (quotas[0].1, quotas[0].2);
    for (label, remaining, reset) in &quotas {
        usage.models.push(ModelQuota { label: label.clone(), remaining: format_percent(remaining * 100.0), resets_in: format_duration(*reset) });
        if *remaining < lowest.0 {
            lowest = (*remaining, *reset);
        }
    }
    usage.current_limit = format_percent(lowest.0 * 100.0);
    usage.time_to_reset_session = format_duration(lowest.1);
    usage
}

impl AgyAgent {
    pub fn usage_with(&self, runner: Runner) -> Usage {
        match runner(vec!["antigravity-usage".into(), "quota".into(), "--json".into()]) {
            Ok(output) => parse_agy_usage(&Json::object(&output)),
            Err(_) => Usage::default(),
        }
    }
}

impl Agent for AgyAgent {
    fn call(&self, model: &str, prompt: &str, args: &[String]) -> Result<String, String> {
        validate(model, prompt)?;
        execute(Self::command(model, prompt, args))
    }

    fn usage(&self) -> Usage {
        self.usage_with(execute)
    }
}

impl Agent for OllamaAgent {
    fn call(&self, model: &str, prompt: &str, args: &[String]) -> Result<String, String> {
        validate(model, prompt)?;
        execute(Self::command(model, prompt, args))
    }

    fn usage(&self) -> Usage {
        Usage { weekly_limit: "100%".into(), current_limit: "100%".into(), time_to_reset_week: NEVER.into(), time_to_reset_session: NEVER.into(), models: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quota(label: &str, remaining: &str, resets_in: &str) -> ModelQuota {
        ModelQuota { label: label.into(), remaining: remaining.into(), resets_in: resets_in.into() }
    }

    #[test]
    fn providers_build_argument_arrays_without_a_shell() {
        let args = vec!["--flag".into()];
        assert_eq!(ClaudeCodeAgent::command("model-a", "hello world", &args), vec!["claude", "-p", "--model", "model-a", "--flag", "hello world"]);
        assert_eq!(CodexAgent::command("model-a", "hello world", &args), vec!["codex", "exec", "--model", "model-a", "--flag", "hello world"]);
        assert_eq!(AgyAgent::command("model-a", "hello world", &args), vec!["agy", "-p", "--model", "model-a", "--flag", "hello world"]);
        assert_eq!(OllamaAgent::command("model-a", "hello world", &args), vec!["ollama", "run", "--flag", "model-a", "hello world"]);
    }

    #[test]
    fn required_inputs_are_explicit() {
        assert_eq!(OllamaAgent.call("", "hello", &[]), Err("model is required".into()));
        assert_eq!(OllamaAgent.call("model", "", &[]), Err("prompt is required".into()));
    }

    #[test]
    fn ollama_is_always_available_because_it_runs_locally() {
        let usage = OllamaAgent.usage();
        assert_eq!(usage.current_limit, "100%");
        assert_eq!(usage.weekly_limit, "100%");
        assert_eq!(usage.time_to_reset_session, "never");
        assert_eq!(usage.time_to_reset_week, "never");
    }

    #[test]
    fn agy_reports_a_quota_for_every_model() {
        let payload = r#"{"models":[{"label":"Gemini 3 Pro","remainingPercentage":0.99,"timeUntilResetMs":3000000},{"label":"Claude Opus 4.6","remainingPercentage":0.42,"timeUntilResetMs":90000}]}"#;
        let usage = parse_agy_usage(&Json::object(payload));
        assert_eq!(usage.models, vec![quota("Gemini 3 Pro", "99%", "50m"), quota("Claude Opus 4.6", "42%", "1m")]);
        assert_eq!(usage.current_limit, "42%");
        assert_eq!(usage.time_to_reset_session, "1m");
        assert_eq!(usage.weekly_limit, "unavailable");
    }

    #[test]
    fn agy_ignores_autocomplete_only_models_because_they_hold_a_separate_quota() {
        let payload = r#"{"models":[{"label":"Gemini 3 Pro","remainingPercentage":0.8,"timeUntilResetMs":60000},{"label":"Gemini 2.5 Flash","remainingPercentage":0.05,"timeUntilResetMs":60000,"isAutocompleteOnly":true}]}"#;
        let usage = parse_agy_usage(&Json::object(payload));
        assert_eq!(usage.models, vec![quota("Gemini 3 Pro", "80%", "1m")]);
        assert_eq!(usage.current_limit, "80%");
    }

    #[test]
    fn agy_keeps_the_most_constrained_quota_for_a_repeated_model_label() {
        let payload = r#"{"models":[{"label":"Gemini 3.1 Pro (High)","modelId":"gemini-pro-agent","remainingPercentage":0.9,"timeUntilResetMs":60000},{"label":"Gemini 3.1 Pro (High)","modelId":"gemini-3.1-pro-high","remainingPercentage":0.4,"timeUntilResetMs":60000}]}"#;
        assert_eq!(parse_agy_usage(&Json::object(payload)).models, vec![quota("Gemini 3.1 Pro (High)", "40%", "1m")]);
    }

    #[test]
    fn agy_usage_survives_a_banner_printed_before_the_json() {
        let output = "\nAntigravity Quota Status\n{\"models\":[{\"label\":\"Gemini 3 Pro\",\"remainingPercentage\":0.25,\"timeUntilResetMs\":60000}]}\n";
        assert_eq!(parse_agy_usage(&Json::object(output)).current_limit, "25%");
    }

    #[test]
    fn agy_usage_is_unavailable_when_antigravity_usage_fails() {
        fn failing(_command: Vec<String>) -> Result<String, String> {
            Err("not installed".into())
        }
        assert_eq!(AgyAgent.usage_with(failing), Usage::default());
    }

    #[test]
    fn claude_reads_both_oauth_windows_as_remaining_capacity() {
        let payload = r#"{"five_hour":{"utilization":23,"resets_at":"2126-08-17T05:00:00+00:00"},"seven_day":{"utilization":41,"resets_at":"2126-08-23T05:00:00+00:00"}}"#;
        let usage = parse_claude_usage(&Json::object(payload));
        assert_eq!(usage.current_limit, "77%");
        assert_eq!(usage.weekly_limit, "59%");
        assert_ne!(usage.time_to_reset_session, "unavailable");
    }

    #[test]
    fn claude_usage_asks_the_keychain_and_never_puts_the_token_in_the_arguments() {
        fn runner(command: Vec<String>) -> Result<String, String> {
            if command[0] == "security" {
                return Ok(r#"{"claudeAiOauth":{"accessToken":"secret-token"}}"#.into());
            }
            assert_eq!(command[..3], ["curl", "-sS", "-K"]);
            let config = fs::read_to_string(&command[3]).expect("the curl config must exist while curl runs");
            assert!(config.contains("secret-token"));
            assert!(!command.join(" ").contains("secret-token"));
            Ok(r#"{"five_hour":{"utilization":10,"resets_at":null}}"#.into())
        }
        unsafe { std::env::remove_var("CLAUDE_CODE_OAUTH_TOKEN") };
        assert_eq!(ClaudeCodeAgent.usage_with(runner).current_limit, "90%");
    }

    #[test]
    fn claude_usage_is_unavailable_without_a_token() {
        fn failing(_command: Vec<String>) -> Result<String, String> {
            Err("no keychain".into())
        }
        unsafe {
            std::env::remove_var("CLAUDE_CODE_OAUTH_TOKEN");
            std::env::set_var("CLAUDE_CONFIG_DIR", std::env::temp_dir().join("agent-sdk-missing"));
        }
        assert_eq!(ClaudeCodeAgent.usage_with(failing), Usage::default());
    }

    #[test]
    fn codex_reports_only_the_windows_the_provider_returns() {
        let payload = r#"{"primary":{"used_percent":52,"window_minutes":10080,"resets_at":4102444800},"secondary":null}"#;
        let usage = parse_codex_usage(&Json::object(payload));
        assert_eq!(usage.weekly_limit, "48%");
        assert_eq!(usage.current_limit, "unavailable");
        assert_eq!(usage.time_to_reset_session, "unavailable");
    }

    #[test]
    fn codex_without_any_window_is_unavailable() {
        assert_eq!(parse_codex_usage(&Json::object(r#"{"primary":null,"secondary":null}"#)), Usage::default());
    }

    #[test]
    fn codex_reads_the_newest_session_that_carries_rate_limits() {
        let home = std::env::temp_dir().join(format!("agent-sdk-codex-{}", std::process::id()));
        let sessions = home.join("sessions").join("2026");
        fs::create_dir_all(&sessions).unwrap();
        fs::write(sessions.join("empty.jsonl"), "{\"payload\":{\"rate_limits\":{\"primary\":null,\"secondary\":null}}}\n").unwrap();
        fs::write(sessions.join("newest.jsonl"), "{\"payload\":{\"rate_limits\":{\"primary\":{\"used_percent\":90,\"window_minutes\":300,\"resets_at\":4102444800}}}}\n").unwrap();
        let limits = latest_rate_limits(&home);
        assert_eq!(limits.get("primary").and_then(|window| window.get("used_percent")).and_then(Json::number), Some(90.0));
        assert_eq!(parse_codex_usage(&limits).current_limit, "10%");
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn an_iso_timestamp_becomes_epoch_milliseconds() {
        assert_eq!(parse_iso_millis("1970-01-01T00:00:00+00:00"), Some(0));
        assert_eq!(parse_iso_millis("2026-08-17T05:00:00Z"), Some(1786942800000));
        assert_eq!(parse_iso_millis("2026-08-17T05:00:00-03:00"), Some(1786953600000));
    }
}
