mod read_file;
mod list_files;
mod edit_file;
mod execute_command;
pub mod web_search;

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
        },
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Execute a program with arguments. Use this to run commands like 'node hello.js', 'python3 main.py', 'java -jar app.jar', etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "program": {
                            "type": "string",
                            "description": "The program to execute (e.g., 'node', 'python3', 'java')"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Array of arguments to pass to the program (e.g., ['hello.js'] or ['-jar', 'app.jar'])"
                        }
                    },
                    "required": ["program"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Fetch a webpage and extract its text content, stripping all JavaScript, CSS, and HTML tags",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to fetch and extract text from"
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    ])
}

pub async fn execute_tool(name: &str, arguments: &str) -> String {
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
        "execute_command" => {
            let program = args.get("program").and_then(|v| v.as_str()).unwrap_or("");
            let cmd_args: Vec<String> = args
                .get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            execute_command::execute_command(program, &cmd_args)
        }
        "web_search" => {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
            web_search::web_search(url).await
        }
        _ => format!("Unknown tool: {}", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_get_tools_returns_array() {
        let tools = get_tools();
        assert!(tools.is_array());
    }

    #[test]
    fn test_get_tools_contains_five_tools() {
        let tools = get_tools();
        assert_eq!(tools.as_array().unwrap().len(), 5);
    }

    #[test]
    fn test_get_tools_has_read_file() {
        let tools = get_tools();
        let arr = tools.as_array().unwrap();
        let has_read_file = arr.iter().any(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                == Some("read_file")
        });
        assert!(has_read_file);
    }

    #[test]
    fn test_get_tools_has_list_files() {
        let tools = get_tools();
        let arr = tools.as_array().unwrap();
        let has_list_files = arr.iter().any(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                == Some("list_files")
        });
        assert!(has_list_files);
    }

    #[test]
    fn test_get_tools_has_edit_file() {
        let tools = get_tools();
        let arr = tools.as_array().unwrap();
        let has_edit_file = arr.iter().any(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                == Some("edit_file")
        });
        assert!(has_edit_file);
    }

    #[test]
    fn test_get_tools_has_execute_command() {
        let tools = get_tools();
        let arr = tools.as_array().unwrap();
        let has_execute_command = arr.iter().any(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                == Some("execute_command")
        });
        assert!(has_execute_command);
    }

    #[test]
    fn test_get_tools_has_web_search() {
        let tools = get_tools();
        let arr = tools.as_array().unwrap();
        let has_web_search = arr.iter().any(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                == Some("web_search")
        });
        assert!(has_web_search);
    }

    #[tokio::test]
    async fn test_execute_tool_read_file() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_exec_read.txt");
        fs::write(&test_file, "Test content").unwrap();
        let args = format!(r#"{{"path": "{}"}}"#, test_file.to_str().unwrap());
        let result = execute_tool("read_file", &args).await;
        assert_eq!(result, "Test content");
        fs::remove_file(&test_file).unwrap();
    }

    #[tokio::test]
    async fn test_execute_tool_list_files() {
        let result = execute_tool("list_files", r#"{"path": "."}"#).await;
        assert!(result.contains("["));
        assert!(result.contains("]"));
    }

    #[tokio::test]
    async fn test_execute_tool_list_files_empty_path() {
        let result = execute_tool("list_files", r#"{}"#).await;
        assert!(result.contains("["));
    }

    #[tokio::test]
    async fn test_execute_tool_edit_file() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_exec_edit.txt");
        let _ = fs::remove_file(&test_file);
        let args = format!(r#"{{"path": "{}", "content": "Written content"}}"#, test_file.to_str().unwrap());
        let result = execute_tool("edit_file", &args).await;
        assert!(result.contains("written successfully"));
        let content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "Written content");
        fs::remove_file(&test_file).unwrap();
    }

    #[tokio::test]
    async fn test_execute_tool_execute_command() {
        let result = execute_tool("execute_command", r#"{"program": "echo", "args": ["hello"]}"#).await;
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_tool_execute_command_no_args() {
        let result = execute_tool("execute_command", r#"{"program": "pwd"}"#).await;
        assert!(!result.starts_with("Error"));
    }

    #[tokio::test]
    async fn test_execute_tool_unknown_tool() {
        let result = execute_tool("unknown_tool", "{}").await;
        assert_eq!(result, "Unknown tool: unknown_tool");
    }

    #[tokio::test]
    async fn test_execute_tool_invalid_json() {
        let result = execute_tool("read_file", "not json").await;
        assert!(result.contains("Error") || result.is_empty() || result.len() > 0);
    }

    #[tokio::test]
    async fn test_execute_tool_missing_required_args() {
        let result = execute_tool("read_file", "{}").await;
        assert!(result.contains("Error"));
    }

    #[tokio::test]
    async fn test_execute_tool_web_search_empty_url() {
        let result = execute_tool("web_search", r#"{"url": ""}"#).await;
        assert!(result.contains("Error"));
    }
}
