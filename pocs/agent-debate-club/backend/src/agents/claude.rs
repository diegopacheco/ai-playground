use super::runner::AgentRunner;

pub fn create_claude_runner() -> AgentRunner {
    AgentRunner::new("claude")
}
