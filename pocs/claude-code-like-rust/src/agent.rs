use reqwest::Client;
use serde_json::json;
use crate::llm::api::call_openai;
use crate::llm::types::Message;
use crate::tools::execute_tool;

pub async fn agent_loop(
    client: &Client,
    api_key: &str,
    messages: &mut Vec<Message>,
) -> Result<String, Box<dyn std::error::Error>> {
    loop {
        let response = call_openai(client, api_key, messages).await?;
        let choice = &response.choices[0];
        let msg = &choice.message;

        messages.push(Message {
            role: msg.role.clone(),
            content: msg.content.clone().map(|s| json!(s)),
            tool_calls: msg.tool_calls.clone(),
            tool_call_id: None,
        });

        if let Some(tool_calls) = &msg.tool_calls {
            if let Some(text) = &msg.content {
                if !text.is_empty() {
                    println!("{}", text);
                }
            }

            for tool_call in tool_calls {
                let name = &tool_call.function.name;
                let args = &tool_call.function.arguments;
                println!("[Tool: {}]", name);
                let result = execute_tool(name, args);
                let truncated = if result.len() > 200 {
                    format!("{}...", &result[..200])
                } else {
                    result.clone()
                };
                println!("Result: {}", truncated);

                messages.push(Message {
                    role: "tool".to_string(),
                    content: Some(json!(result)),
                    tool_calls: None,
                    tool_call_id: Some(tool_call.id.clone()),
                });
            }
        } else {
            return Ok(msg.content.clone().unwrap_or_default());
        }

        if choice.finish_reason == Some("stop".to_string()) {
            return Ok(msg.content.clone().unwrap_or_default());
        }
    }
}
