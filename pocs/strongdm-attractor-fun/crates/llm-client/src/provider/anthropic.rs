use crate::error::{classify_status, LlmError};
use crate::types::*;
use reqwest::Client;

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.anthropic.com/v1".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    fn build_messages(&self, request: &Request) -> Vec<serde_json::Value> {
        let mut msgs = Vec::new();
        for msg in &request.messages {
            if msg.role == Role::System {
                continue;
            }
            let role = match msg.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "assistant",
                Role::System => continue,
            };
            let content: Vec<serde_json::Value> = msg
                .content
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text
                    }),
                    ContentPart::ToolUse { id, name, input } => serde_json::json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": input
                    }),
                    ContentPart::ToolResult {
                        tool_use_id,
                        content,
                    } => serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content
                    }),
                })
                .collect();
            msgs.push(serde_json::json!({
                "role": role,
                "content": content
            }));
        }
        msgs
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                })
            })
            .collect()
    }

    pub async fn complete(&self, request: &Request) -> Result<Response, LlmError> {
        let mut body = serde_json::json!({
            "model": request.model,
            "messages": self.build_messages(request),
            "max_tokens": request.max_tokens,
        });
        if let Some(sys) = &request.system {
            body["system"] = serde_json::json!(sys);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if !request.tools.is_empty() {
            body["tools"] = serde_json::json!(self.build_tools(&request.tools));
        }

        let resp = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();
        let text = resp.text().await?;
        if status != 200 {
            return Err(classify_status(status, &text));
        }

        let json: serde_json::Value = serde_json::from_str(&text)?;
        self.parse_response(&json)
    }

    fn parse_response(&self, json: &serde_json::Value) -> Result<Response, LlmError> {
        let mut content = Vec::new();
        if let Some(parts) = json["content"].as_array() {
            for part in parts {
                match part["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = part["text"].as_str() {
                            content.push(ContentPart::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                    Some("tool_use") => {
                        let id = part["id"].as_str().unwrap_or("").to_string();
                        let name = part["name"].as_str().unwrap_or("").to_string();
                        let input = part["input"].clone();
                        content.push(ContentPart::ToolUse { id, name, input });
                    }
                    _ => {}
                }
            }
        }

        let usage = Usage {
            input_tokens: json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
        };

        let stop_reason = json["stop_reason"].as_str().map(|s| s.to_string());
        let model = json["model"].as_str().unwrap_or("").to_string();

        Ok(Response {
            content,
            usage,
            stop_reason,
            model,
        })
    }
}
