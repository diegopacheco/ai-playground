use crate::models::ModelResult;
use super::execute_claude;

pub async fn analyze(prompt: &str) -> ModelResult {
    execute_claude("claude-opus-4-5-20251101", prompt).await
}
