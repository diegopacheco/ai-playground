use super::runner::AgentRunner;

pub fn create_copilot_runner() -> AgentRunner {
    AgentRunner::new("copilot")
}
