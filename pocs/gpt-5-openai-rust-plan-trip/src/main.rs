use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{self, Write};

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("AI Trip Planner (ai-agent-planner-rs)");

    let destination = prompt("Enter destination city (e.g., New York): ")?;
    let days_num: u32 = loop {
        let raw = prompt("How many days will you stay there?: ")?;
        match raw.parse::<u32>() {
            Ok(n) if n > 0 => break n,
            _ => {
                eprintln!("Please enter a positive whole number for days.");
                continue;
            }
        }
    };

    let system = "You are a helpful travel planning assistant. Create detailed, practical, and engaging day-by-day itineraries with local tips, must-try food, and estimated times. Keep it concise but useful.";
    let user = format!(
        "Plan a day-by-day travel itinerary for a trip to {destination} for {days_num} days. Include:
- Morning/afternoon/evening suggestions each day
- A mix of landmarks and hidden gems
- Food/coffee/dessert ideas near activities
- Public transit or walking tips when helpful
- Rainy-day alternatives if relevant
Keep it structured and friendly."
    );

    let api_key = env::var("OPENAI_API_KEY")
        .context("Missing OPENAI_API_KEY environment variable. Export it in your shell.")?;

    let client = reqwest::Client::new();

    let url = "https://api.openai.com/v1/chat/completions";
    let req_body = ChatRequest {
        model: "gpt-5",
        messages: vec![
            Message {
                role: "system",
                content: system,
            },
            Message {
                role: "user",
                content: &user,
            },
        ],
    };

    let res = client
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, format!("Bearer {}", api_key))
        .json(&req_body)
        .send()
        .await
        .context("Failed to send request to OpenAI API")?;

    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI API error: {}\n{}", status, text);
    }

    let body: ChatResponse = res.json().await.context("Failed to parse OpenAI response")?;
    if let Some(choice) = body.choices.first() {
        println!(
            "\nYour plan for {} ({} days):\n\n{}",
            destination,
            days_num,
            choice.message.content.trim()
        );
    } else {
        println!("No plan generated. Try again.");
    }

    Ok(())
}

fn prompt(msg: &str) -> Result<String> {
    print!("{}", msg);
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).context("Failed to read input")?;
    Ok(input.trim().to_string())
}
