use reqwest::Client;
use serde_json::json;
use crate::llm::api::call_openai;
use crate::llm::types::{Message, ResponseMessage, ToolCall};
use crate::tools::execute_tool;

pub const TRUNCATE_LENGTH: usize = 200;

pub fn truncate_result(result: &str) -> String {
    if result.len() > TRUNCATE_LENGTH {
        format!("{}...", &result[..TRUNCATE_LENGTH])
    } else {
        result.to_string()
    }
}

pub fn create_assistant_message(msg: &ResponseMessage) -> Message {
    Message {
        role: msg.role.clone(),
        content: msg.content.clone().map(|s| json!(s)),
        tool_calls: msg.tool_calls.clone(),
        tool_call_id: None,
    }
}

pub fn create_tool_result_message(tool_call_id: &str, result: &str) -> Message {
    Message {
        role: "tool".to_string(),
        content: Some(json!(result)),
        tool_calls: None,
        tool_call_id: Some(tool_call_id.to_string()),
    }
}

pub async fn process_tool_call(tool_call: &ToolCall) -> (String, String, String) {
    let name = tool_call.function.name.clone();
    let args = tool_call.function.arguments.clone();
    let result = execute_tool(&name, &args).await;
    (name, args, result)
}

pub async fn agent_loop(
    client: &Client,
    api_key: &str,
    messages: &mut Vec<Message>,
) -> Result<String, Box<dyn std::error::Error>> {
    loop {
        let response = call_openai(client, api_key, messages).await?;
        let choice = &response.choices[0];
        let msg = &choice.message;

        messages.push(create_assistant_message(msg));

        if let Some(tool_calls) = &msg.tool_calls {
            if let Some(text) = &msg.content {
                if !text.is_empty() {
                    println!("{}", text);
                }
            }

            for tool_call in tool_calls {
                let (name, _, result) = process_tool_call(tool_call).await;
                println!("[Tool: {}]", name);
                println!("Result: {}", truncate_result(&result));

                messages.push(create_tool_result_message(&tool_call.id, &result));
            }
        } else {
            return Ok(msg.content.clone().unwrap_or_default());
        }

        if choice.finish_reason == Some("stop".to_string()) {
            return Ok(msg.content.clone().unwrap_or_default());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{FunctionCall, ToolCall};

    #[test]
    fn test_truncate_length_constant() {
        assert_eq!(TRUNCATE_LENGTH, 200);
    }

    #[test]
    fn test_truncate_result_short_string() {
        let result = truncate_result("Hello");
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_truncate_result_exact_length() {
        let text = "a".repeat(200);
        let result = truncate_result(&text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_truncate_result_long_string() {
        let text = "a".repeat(300);
        let result = truncate_result(&text);
        assert_eq!(result.len(), 203);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_result_empty_string() {
        let result = truncate_result("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_create_assistant_message() {
        let response_msg = ResponseMessage {
            role: "assistant".to_string(),
            content: Some("Hello, user!".to_string()),
            tool_calls: None,
        };
        let msg = create_assistant_message(&response_msg);
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.unwrap().as_str().unwrap(), "Hello, user!");
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn test_create_assistant_message_with_tool_calls() {
        let tool_calls = vec![ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"path": "test.txt"}"#.to_string(),
            },
        }];
        let response_msg = ResponseMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
        };
        let msg = create_assistant_message(&response_msg);
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_create_assistant_message_no_content() {
        let response_msg = ResponseMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
        };
        let msg = create_assistant_message(&response_msg);
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_create_tool_result_message() {
        let msg = create_tool_result_message("call_abc", "File content here");
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.content.unwrap().as_str().unwrap(), "File content here");
        assert!(msg.tool_calls.is_none());
        assert_eq!(msg.tool_call_id.unwrap(), "call_abc");
    }

    #[test]
    fn test_create_tool_result_message_empty_result() {
        let msg = create_tool_result_message("call_xyz", "");
        assert_eq!(msg.content.unwrap().as_str().unwrap(), "");
    }

    #[tokio::test]
    async fn test_process_tool_call_read_file() {
        use std::env;
        use std::fs;
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_process_tool_call.txt");
        fs::write(&test_file, "Test content").unwrap();
        let tool_call = ToolCall {
            id: "call_test".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: format!(r#"{{"path": "{}"}}"#, test_file.to_str().unwrap()),
            },
        };
        let (name, _, result) = process_tool_call(&tool_call).await;
        assert_eq!(name, "read_file");
        assert_eq!(result, "Test content");
        fs::remove_file(&test_file).unwrap();
    }

    #[tokio::test]
    async fn test_process_tool_call_list_files() {
        let tool_call = ToolCall {
            id: "call_list".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "list_files".to_string(),
                arguments: r#"{"path": "."}"#.to_string(),
            },
        };
        let (name, _, result) = process_tool_call(&tool_call).await;
        assert_eq!(name, "list_files");
        assert!(result.contains("["));
    }

    #[tokio::test]
    async fn test_process_tool_call_execute_command() {
        let tool_call = ToolCall {
            id: "call_exec".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "execute_command".to_string(),
                arguments: r#"{"program": "echo", "args": ["hello"]}"#.to_string(),
            },
        };
        let (name, _, result) = process_tool_call(&tool_call).await;
        assert_eq!(name, "execute_command");
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn test_process_tool_call_unknown_tool() {
        let tool_call = ToolCall {
            id: "call_unknown".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "unknown_tool".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let (name, _, result) = process_tool_call(&tool_call).await;
        assert_eq!(name, "unknown_tool");
        assert!(result.contains("Unknown tool"));
    }
}
