pub mod runner;
pub mod claude;
pub mod gemini;

use runner::AgentRunner;

pub fn get_runner(agent_type: &str) -> AgentRunner {
    AgentRunner::new(agent_type)
}

pub fn get_available_agents() -> Vec<(String, String)> {
    vec![
        ("claude".to_string(), "Claude".to_string()),
        ("gemini".to_string(), "Gemini".to_string()),
    ]
}
