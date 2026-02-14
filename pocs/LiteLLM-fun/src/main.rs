use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageResponse,
}

#[derive(Deserialize)]
struct MessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

fn main() {
    let client = reqwest::blocking::Client::new();
    let request = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }],
    };

    let response = client
        .post("http://0.0.0.0:4000/v1/chat/completions")
        .header("Authorization", "Bearer sk-1234")
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .expect("Failed to send request");

    let chat: ChatResponse = response.json().expect("Failed to parse response");
    println!("{}", chat.choices[0].message.content);
}
