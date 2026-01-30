use crate::models::ModelResult;
use super::execute_copilot;

pub async fn analyze(prompt: &str) -> ModelResult {
    execute_copilot("claude-sonnet-4", prompt).await
}
