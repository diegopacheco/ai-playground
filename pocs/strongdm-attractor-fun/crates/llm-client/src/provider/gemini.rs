use crate::error::{classify_status, LlmError};
use crate::types::*;
use reqwest::Client;

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    fn build_contents(&self, request: &Request) -> Vec<serde_json::Value> {
        let mut contents = Vec::new();
        for msg in &request.messages {
            if msg.role == Role::System {
                continue;
            }
            let role = match msg.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "model",
                Role::System => continue,
            };
            let parts: Vec<serde_json::Value> = msg
                .content
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => serde_json::json!({ "text": text }),
                    ContentPart::ToolUse { name, input, .. } => serde_json::json!({
                        "functionCall": {
                            "name": name,
                            "args": input
                        }
                    }),
                    ContentPart::ToolResult {
                        tool_use_id: _,
                        content,
                    } => serde_json::json!({
                        "functionResponse": {
                            "name": "tool",
                            "response": { "result": content }
                        }
                    }),
                })
                .collect();
            contents.push(serde_json::json!({
                "role": role,
                "parts": parts
            }));
        }
        contents
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        let declarations: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                })
            })
            .collect();
        vec![serde_json::json!({ "functionDeclarations": declarations })]
    }

    pub async fn complete(&self, request: &Request) -> Result<Response, LlmError> {
        let mut body = serde_json::json!({
            "contents": self.build_contents(request),
        });
        if let Some(sys) = &request.system {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{ "text": sys }]
            });
        }
        let mut config = serde_json::json!({
            "maxOutputTokens": request.max_tokens,
        });
        if let Some(temp) = request.temperature {
            config["temperature"] = serde_json::json!(temp);
        }
        body["generationConfig"] = config;
        if !request.tools.is_empty() {
            body["tools"] = serde_json::json!(self.build_tools(&request.tools));
        }

        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, request.model, self.api_key
        );
        let resp = self
            .client
            .post(&url)
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
        let candidate = json["candidates"]
            .get(0)
            .ok_or_else(|| LlmError::Parse("no candidates in response".into()))?;
        let parts = candidate["content"]["parts"]
            .as_array()
            .ok_or_else(|| LlmError::Parse("no parts in candidate".into()))?;

        let mut content = Vec::new();
        for part in parts {
            if let Some(text) = part["text"].as_str() {
                content.push(ContentPart::Text {
                    text: text.to_string(),
                });
            }
            if part.get("functionCall").is_some() {
                let name = part["functionCall"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let input = part["functionCall"]["args"].clone();
                let id = format!("gemini_{}", name);
                content.push(ContentPart::ToolUse { id, name, input });
            }
        }

        let usage = Usage {
            input_tokens: json["usageMetadata"]["promptTokenCount"]
                .as_u64()
                .unwrap_or(0) as u32,
            output_tokens: json["usageMetadata"]["candidatesTokenCount"]
                .as_u64()
                .unwrap_or(0) as u32,
        };

        let stop_reason = candidate["finishReason"].as_str().map(|s| s.to_string());

        Ok(Response {
            content,
            usage,
            stop_reason,
            model: String::new(),
        })
    }
}
