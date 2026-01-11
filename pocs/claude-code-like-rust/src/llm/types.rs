use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_serialize_user() {
        let msg = Message {
            role: "user".to_string(),
            content: Some(json!("Hello")),
            tool_calls: None,
            tool_call_id: None,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        assert!(json_str.contains("user"));
        assert!(json_str.contains("Hello"));
    }

    #[test]
    fn test_message_serialize_skips_none_fields() {
        let msg = Message {
            role: "user".to_string(),
            content: Some(json!("Hello")),
            tool_calls: None,
            tool_call_id: None,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        assert!(!json_str.contains("tool_calls"));
        assert!(!json_str.contains("tool_call_id"));
    }

    #[test]
    fn test_message_serialize_with_tool_calls() {
        let msg = Message {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path": "test.txt"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        assert!(json_str.contains("call_123"));
        assert!(json_str.contains("read_file"));
    }

    #[test]
    fn test_message_deserialize() {
        let json_str = r#"{"role": "user", "content": "Hello"}"#;
        let msg: Message = serde_json::from_str(json_str).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.unwrap().as_str().unwrap(), "Hello");
    }

    #[test]
    fn test_message_clone() {
        let msg = Message {
            role: "user".to_string(),
            content: Some(json!("Test")),
            tool_calls: None,
            tool_call_id: None,
        };
        let cloned = msg.clone();
        assert_eq!(cloned.role, msg.role);
    }

    #[test]
    fn test_tool_call_serialize() {
        let tool_call = ToolCall {
            id: "call_abc".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "list_files".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let json_str = serde_json::to_string(&tool_call).unwrap();
        assert!(json_str.contains("call_abc"));
        assert!(json_str.contains("\"type\":\"function\""));
        assert!(json_str.contains("list_files"));
    }

    #[test]
    fn test_tool_call_deserialize() {
        let json_str = r#"{"id": "call_xyz", "type": "function", "function": {"name": "test", "arguments": "{}"}}"#;
        let tool_call: ToolCall = serde_json::from_str(json_str).unwrap();
        assert_eq!(tool_call.id, "call_xyz");
        assert_eq!(tool_call.call_type, "function");
        assert_eq!(tool_call.function.name, "test");
    }

    #[test]
    fn test_function_call_serialize() {
        let func = FunctionCall {
            name: "edit_file".to_string(),
            arguments: r#"{"path": "a.txt", "content": "hello"}"#.to_string(),
        };
        let json_str = serde_json::to_string(&func).unwrap();
        assert!(json_str.contains("edit_file"));
        assert!(json_str.contains("a.txt"));
    }

    #[test]
    fn test_function_call_deserialize() {
        let json_str = r#"{"name": "read_file", "arguments": "{\"path\": \"test.txt\"}"}"#;
        let func: FunctionCall = serde_json::from_str(json_str).unwrap();
        assert_eq!(func.name, "read_file");
        assert!(func.arguments.contains("path"));
    }

    #[test]
    fn test_api_response_deserialize() {
        let json_str = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }]
        }"#;
        let response: ApiResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.role, "assistant");
        assert_eq!(response.choices[0].message.content.as_ref().unwrap(), "Hello!");
        assert_eq!(response.choices[0].finish_reason.as_ref().unwrap(), "stop");
    }

    #[test]
    fn test_api_response_with_tool_calls() {
        let json_str = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\": \"test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let response: ApiResponse = serde_json::from_str(json_str).unwrap();
        assert!(response.choices[0].message.tool_calls.is_some());
        let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read_file");
    }

    #[test]
    fn test_choice_finish_reason_none() {
        let json_str = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Working..."
                }
            }]
        }"#;
        let response: ApiResponse = serde_json::from_str(json_str).unwrap();
        assert!(response.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_response_message_deserialize() {
        let json_str = r#"{"role": "assistant", "content": "Test response"}"#;
        let msg: ResponseMessage = serde_json::from_str(json_str).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.unwrap(), "Test response");
        assert!(msg.tool_calls.is_none());
    }
}
