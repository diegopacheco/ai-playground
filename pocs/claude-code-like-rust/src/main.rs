use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

fn read_file(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => format!("Error reading file: {}", e),
    }
}

fn list_files(path: &str) -> String {
    let dir_path = if path.is_empty() { "." } else { path };
    match fs::read_dir(dir_path) {
        Ok(entries) => {
            let mut result = Vec::new();
            for entry in entries.flatten() {
                let file_type = if entry.path().is_dir() { "directory" } else { "file" };
                result.push(json!({
                    "name": entry.file_name().to_string_lossy(),
                    "type": file_type
                }));
            }
            serde_json::to_string_pretty(&result).unwrap_or_default()
        }
        Err(e) => format!("Error listing directory: {}", e),
    }
}

fn edit_file(path: &str, content: &str) -> String {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                return format!("Error creating directories: {}", e);
            }
        }
    }
    match fs::write(path, content) {
        Ok(_) => format!("File '{}' written successfully", path),
        Err(e) => format!("Error writing file: {}", e),
    }
}

fn execute_tool(name: &str, arguments: &str) -> String {
    let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));
    match name {
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            read_file(path)
        }
        "list_files" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            list_files(path)
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            edit_file(path, content)
        }
        _ => format!("Unknown tool: {}", name),
    }
}

fn get_tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file at the specified path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories at the specified path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to list files from (defaults to current directory)"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Create or overwrite a file with the provided content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path where the file should be created/modified"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ])
}

async fn call_openai(
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

async fn agent_loop(
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

#[tokio::main]
async fn main() {
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let client = Client::new();
    let mut messages: Vec<Message> = vec![Message {
        role: "system".to_string(),
        content: Some(json!("You are a helpful coding assistant. You can read files, list directories, and edit files to help users with their coding tasks.")),
        tool_calls: None,
        tool_call_id: None,
    }];

    println!("Claude Code (Rust) - Type 'quit' or 'exit' to exit");
    println!("---------------------------------------------------");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: Some(json!(input)),
            tool_calls: None,
            tool_call_id: None,
        });

        match agent_loop(&client, &api_key, &mut messages).await {
            Ok(response) => {
                if !response.is_empty() {
                    println!("{}", response);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
        println!();
    }
}
