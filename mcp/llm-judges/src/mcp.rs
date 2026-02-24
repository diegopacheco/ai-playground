use serde_json::{json, Value};
use crate::tools;

pub async fn handle_message(msg: &Value) -> Option<Value> {
    let id = msg.get("id");
    let method = msg["method"].as_str().unwrap_or("");

    match method {
        "initialize" => {
            Some(response(id, json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "llm-judges",
                    "version": "0.1.0"
                }
            })))
        }
        "notifications/initialized" => None,
        "tools/list" => {
            let defs = tools::tool_definitions();
            Some(response(id, defs))
        }
        "tools/call" => {
            let params = &msg["params"];
            let name = params["name"].as_str().unwrap_or("");
            let args = &params["arguments"];
            let result = tools::handle_tool_call(name, args).await;
            Some(response(id, result))
        }
        "ping" => {
            Some(response(id, json!({})))
        }
        _ => {
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Method not found: {}", method)
                }
            }))
        }
    }
}

fn response(id: Option<&Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}
