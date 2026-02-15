use crate::error::{classify_status, LlmError};
use crate::types::*;
use reqwest::Client;

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    fn build_messages(&self, request: &Request) -> Vec<serde_json::Value> {
        let mut msgs = Vec::new();
        if let Some(sys) = &request.system {
            msgs.push(serde_json::json!({
                "role": "system",
                "content": sys
            }));
        }
        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };
            if msg.content.len() == 1 {
                match &msg.content[0] {
                    ContentPart::Text { text } => {
                        msgs.push(serde_json::json!({
                            "role": role,
                            "content": text
                        }));
                    }
                    ContentPart::ToolUse { id, name, input } => {
                        msgs.push(serde_json::json!({
                            "role": "assistant",
                            "tool_calls": [{
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": serde_json::to_string(input).unwrap_or_default()
                                }
                            }]
                        }));
                    }
                    ContentPart::ToolResult { tool_use_id, content } => {
                        msgs.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": content
                        }));
                    }
                }
            } else {
                let mut tool_calls = Vec::new();
                let mut text_parts = Vec::new();
                let mut tool_results = Vec::new();
                for part in &msg.content {
                    match part {
                        ContentPart::Text { text } => text_parts.push(text.clone()),
                        ContentPart::ToolUse { id, name, input } => {
                            tool_calls.push(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": serde_json::to_string(input).unwrap_or_default()
                                }
                            }));
                        }
                        ContentPart::ToolResult { tool_use_id, content } => {
                            tool_results.push((tool_use_id.clone(), content.clone()));
                        }
                    }
                }
                if !text_parts.is_empty() || !tool_calls.is_empty() {
                    let mut m = serde_json::json!({ "role": role });
                    if !text_parts.is_empty() {
                        m["content"] = serde_json::json!(text_parts.join(""));
                    }
                    if !tool_calls.is_empty() {
                        m["tool_calls"] = serde_json::json!(tool_calls);
                    }
                    msgs.push(m);
                }
                for (tid, content) in tool_results {
                    msgs.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": content
                    }));
                }
            }
        }
        msgs
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
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
        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if !request.tools.is_empty() {
            body["tools"] = serde_json::json!(self.build_tools(&request.tools));
        }

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
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
        let choice = json["choices"]
            .get(0)
            .ok_or_else(|| LlmError::Parse("no choices in response".into()))?;
        let message = &choice["message"];

        let mut content = Vec::new();
        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                content.push(ContentPart::Text {
                    text: text.to_string(),
                });
            }
        }
        if let Some(calls) = message["tool_calls"].as_array() {
            for call in calls {
                let id = call["id"].as_str().unwrap_or("").to_string();
                let name = call["function"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let args_str = call["function"]["arguments"].as_str().unwrap_or("{}");
                let input: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                content.push(ContentPart::ToolUse { id, name, input });
            }
        }

        let usage = Usage {
            input_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        };

        let stop_reason = choice["finish_reason"].as_str().map(|s| s.to_string());
        let model = json["model"].as_str().unwrap_or("").to_string();

        Ok(Response {
            content,
            usage,
            stop_reason,
            model,
        })
    }
}
