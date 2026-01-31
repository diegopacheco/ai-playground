use super::runner::AgentRunner;

pub fn create_gemini_runner() -> AgentRunner {
    AgentRunner::new("gemini")
}
