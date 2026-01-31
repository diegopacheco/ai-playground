pub mod runner;
pub mod claude;
pub mod gemini;
pub mod copilot;
pub mod codex;

use runner::AgentRunner;

pub fn get_runner(agent_type: &str) -> AgentRunner {
    match agent_type {
        "claude" => claude::create_claude_runner(),
        "gemini" => gemini::create_gemini_runner(),
        "copilot" => copilot::create_copilot_runner(),
        "codex" => codex::create_codex_runner(),
        _ => claude::create_claude_runner(),
    }
}

pub fn get_available_agents() -> Vec<(String, String)> {
    vec![
        ("claude".to_string(), "Claude".to_string()),
        ("gemini".to_string(), "Gemini".to_string()),
        ("copilot".to_string(), "Copilot".to_string()),
        ("codex".to_string(), "Codex".to_string()),
    ]
}
