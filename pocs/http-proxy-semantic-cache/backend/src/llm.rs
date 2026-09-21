use crate::agent_sdk::{Agent, ClaudeCodeAgent};

const INSTRUCTIONS: &str = "Answer the question directly and concisely in plain text.";

pub trait Llm: Send + Sync {
    fn answer(&self, question: &str) -> Result<String, String>;
}

pub struct ClaudeLlm {
    model: String,
}

impl ClaudeLlm {
    pub fn new(model: &str) -> Self {
        Self { model: model.to_string() }
    }
}

impl Llm for ClaudeLlm {
    fn answer(&self, question: &str) -> Result<String, String> {
        let args = vec!["--append-system-prompt".to_string(), INSTRUCTIONS.to_string()];
        let text = ClaudeCodeAgent.call(&self.model, question, &args)?;
        let text = text.trim();
        if text.is_empty() {
            return Err("Claude Code returned an empty answer".into());
        }
        Ok(text.to_string())
    }
}
