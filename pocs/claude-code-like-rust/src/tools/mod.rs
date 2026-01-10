mod read_file;
mod list_files;
mod edit_file;

use serde_json::{json, Value};

pub fn get_tools() -> Value {
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

pub fn execute_tool(name: &str, arguments: &str) -> String {
    let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));
    match name {
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            read_file::read_file(path)
        }
        "list_files" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            list_files::list_files(path)
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            edit_file::edit_file(path, content)
        }
        _ => format!("Unknown tool: {}", name),
    }
}
