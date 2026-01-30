use crate::models::ModelResult;
use super::execute_codex;

pub async fn analyze(prompt: &str) -> ModelResult {
    execute_codex("gpt-5.2-codex", prompt).await
}
