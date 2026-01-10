use reqwest::Client;
use serde_json::json;
use crate::llm::types::{ApiResponse, Message};
use crate::tools::get_tools;

pub async fn call_openai(
    client: &Client,
    api_key: &str,
    messages: &[Message],
) -> Result<ApiResponse, reqwest::Error> {
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": "gpt-4o",
            "messages": messages,
            "tools": get_tools(),
            "tool_choice": "auto"
        }))
        .send()
        .await?
        .json::<ApiResponse>()
        .await?;
    Ok(response)
}
