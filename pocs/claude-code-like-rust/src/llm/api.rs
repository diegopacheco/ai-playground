use reqwest::Client;
use serde_json::json;
use crate::llm::types::{ApiResponse, Message};
use crate::tools::get_tools;

pub const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
pub const MODEL: &str = "gpt-4o";

pub fn build_request_body(messages: &[Message]) -> serde_json::Value {
    json!({
        "model": MODEL,
        "messages": messages,
        "tools": get_tools(),
        "tool_choice": "auto"
    })
}

pub async fn call_openai(
    client: &Client,
    api_key: &str,
    messages: &[Message],
) -> Result<ApiResponse, reqwest::Error> {
    let response = client
        .post(OPENAI_API_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&build_request_body(messages))
        .send()
        .await?
        .json::<ApiResponse>()
        .await?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::Message;
    use serde_json::json;

    #[test]
    fn test_openai_api_url() {
        assert_eq!(OPENAI_API_URL, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_model_constant() {
        assert_eq!(MODEL, "gpt-4o");
    }

    #[test]
    fn test_build_request_body_contains_model() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(json!("Hello")),
            tool_calls: None,
            tool_call_id: None,
        }];
        let body = build_request_body(&messages);
        assert_eq!(body["model"], "gpt-4o");
    }

    #[test]
    fn test_build_request_body_contains_messages() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(json!("Test message")),
            tool_calls: None,
            tool_call_id: None,
        }];
        let body = build_request_body(&messages);
        assert!(body["messages"].is_array());
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_build_request_body_contains_tools() {
        let messages = vec![];
        let body = build_request_body(&messages);
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn test_build_request_body_contains_tool_choice() {
        let messages = vec![];
        let body = build_request_body(&messages);
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn test_build_request_body_multiple_messages() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Some(json!("You are a helpful assistant")),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(json!("Hello")),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Some(json!("Hi there!")),
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let body = build_request_body(&messages);
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_build_request_body_empty_messages() {
        let messages: Vec<Message> = vec![];
        let body = build_request_body(&messages);
        assert!(body["messages"].is_array());
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_build_request_body_with_tool_message() {
        let messages = vec![Message {
            role: "tool".to_string(),
            content: Some(json!("Tool result")),
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        }];
        let body = build_request_body(&messages);
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "call_123");
    }

    #[test]
    fn test_build_request_body_is_valid_json() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(json!("Hello")),
            tool_calls: None,
            tool_call_id: None,
        }];
        let body = build_request_body(&messages);
        let json_str = serde_json::to_string(&body);
        assert!(json_str.is_ok());
    }
}
