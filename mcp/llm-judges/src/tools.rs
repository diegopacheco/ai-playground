use serde_json::{json, Value};
use crate::engine;
use crate::judges;

pub fn tool_definitions() -> Value {
    json!({
        "tools": [
            {
                "name": "judge",
                "description": "Send content to all 4 LLM judges (Claude, Codex, Copilot, Gemini) for evaluation. Use this to fact-check, validate, or judge any content.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to be judged"
                        },
                        "criteria": {
                            "type": "string",
                            "description": "Specific criteria to judge against. Defaults to general fact-checking."
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "judge_pick",
                "description": "Send content to selected LLM judges for evaluation. Choose which judges to use.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to be judged"
                        },
                        "criteria": {
                            "type": "string",
                            "description": "Specific criteria to judge against. Defaults to general fact-checking."
                        },
                        "judges": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "List of judges to use. Options: claude, codex, copilot, gemini"
                        }
                    },
                    "required": ["content", "judges"]
                }
            },
            {
                "name": "list_judges",
                "description": "List all available LLM judges and their CLI commands.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    })
}

pub async fn handle_tool_call(name: &str, args: &Value) -> Value {
    match name {
        "judge" => {
            let content = args["content"].as_str().unwrap_or("");
            let criteria = args["criteria"].as_str();
            let all_judges: Vec<&str> = judges::get_available_judges();
            let result = engine::run_judges(content, criteria, &all_judges).await;
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&result).unwrap()
                }]
            })
        }
        "judge_pick" => {
            let content = args["content"].as_str().unwrap_or("");
            let criteria = args["criteria"].as_str();
            let judge_list: Vec<&str> = match args["judges"].as_array() {
                Some(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
                None => judges::get_available_judges(),
            };
            let result = engine::run_judges(content, criteria, &judge_list).await;
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&result).unwrap()
                }]
            })
        }
        "list_judges" => {
            let available = judges::get_available_judges();
            let judge_info: Vec<Value> = available.iter().map(|&name| {
                json!({
                    "name": name,
                    "cli": name,
                    "available": true
                })
            }).collect();
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&json!({ "judges": judge_info })).unwrap()
                }]
            })
        }
        _ => {
            json!({
                "content": [{
                    "type": "text",
                    "text": format!("Unknown tool: {}", name)
                }],
                "isError": true
            })
        }
    }
}
