use super::runner::AgentRunner;

pub fn create_codex_runner() -> AgentRunner {
    AgentRunner::new("codex")
}
