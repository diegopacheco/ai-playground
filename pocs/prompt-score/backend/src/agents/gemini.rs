use crate::models::ModelResult;
use super::execute_gemini;

pub async fn analyze(prompt: &str) -> ModelResult {
    execute_gemini(prompt).await
}
