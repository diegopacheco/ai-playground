pub mod runner;
pub mod claude;
pub mod gemini;
pub mod copilot;
pub mod codex;

use runner::AgentRunner;

pub fn get_runner(engine: &str) -> AgentRunner {
    AgentRunner::new(engine)
}

pub fn get_available_engines() -> Vec<(String, String)> {
    vec![
        ("claude/opus".to_string(), "Claude Opus".to_string()),
        ("claude/sonnet".to_string(), "Claude Sonnet".to_string()),
        ("codex/gpt-5-2-codex".to_string(), "Codex GPT-5.2".to_string()),
        ("gemini/gemini-3-0".to_string(), "Gemini 3.0".to_string()),
        ("copilot/sonnet".to_string(), "Copilot Sonnet".to_string()),
        ("copilot/opus".to_string(), "Copilot Opus".to_string()),
    ]
}
