mod llm;
mod tools;
mod agent;

use reqwest::Client;
use serde_json::json;
use std::env;
use std::io::{self, Write};
use crate::llm::types::Message;
use crate::agent::agent_loop;

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

    println!("---------------------------------------------------");
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
