use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

const CLAUDE_MODEL: &str = "claude-opus-4-5-20251101";
const TIMEOUT_SECS: u64 = 120;

pub async fn call_claude(prompt: &str) -> Result<String, String> {
    let result = timeout(
        Duration::from_secs(TIMEOUT_SECS),
        Command::new("claude")
            .arg("-p")
            .arg(prompt)
            .arg("--model")
            .arg(CLAUDE_MODEL)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .map_err(|e| format!("Failed to parse output: {}", e))
            } else {
                Err("Claude command failed".to_string())
            }
        }
        Ok(Err(e)) => Err(format!("Failed to execute claude: {}", e)),
        Err(_) => Err("Claude request timed out".to_string()),
    }
}

pub async fn generate_training(topic: &str) -> Result<crate::models::TrainingContent, String> {
    let prompt = format!(
        r#"You are an expert trainer. Create a comprehensive training module about: {}

Generate exactly 5 topics with detailed content for each. Each topic should be educational and practical.

Respond ONLY with valid JSON in this exact format, nothing else:
{{
  "title": "Training title here",
  "topics": [
    {{"id": "1", "title": "Topic 1 Title", "content": "Detailed content for topic 1 with practical information and tips. Make it at least 3-4 paragraphs."}},
    {{"id": "2", "title": "Topic 2 Title", "content": "Detailed content for topic 2..."}},
    {{"id": "3", "title": "Topic 3 Title", "content": "Detailed content for topic 3..."}},
    {{"id": "4", "title": "Topic 4 Title", "content": "Detailed content for topic 4..."}},
    {{"id": "5", "title": "Topic 5 Title", "content": "Detailed content for topic 5..."}}
  ]
}}"#,
        topic
    );

    let response = call_claude(&prompt).await?;
    parse_json_from_response(&response)
}

pub async fn generate_quiz(training_content: &crate::models::TrainingContent) -> Result<crate::models::Quiz, String> {
    let topics_summary: String = training_content
        .topics
        .iter()
        .map(|t| format!("{}: {}", t.title, &t.content[..t.content.len().min(200)]))
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        r#"Based on this training content about "{}":

{}

Generate a quiz with exactly 10 multiple choice questions. Each question should have 4 options with one correct answer.

Respond ONLY with valid JSON in this exact format, nothing else:
{{
  "questions": [
    {{"id": 1, "question": "Question text here?", "options": ["Option A", "Option B", "Option C", "Option D"], "correct_index": 0}},
    {{"id": 2, "question": "Question 2?", "options": ["A", "B", "C", "D"], "correct_index": 1}},
    {{"id": 3, "question": "Question 3?", "options": ["A", "B", "C", "D"], "correct_index": 2}},
    {{"id": 4, "question": "Question 4?", "options": ["A", "B", "C", "D"], "correct_index": 0}},
    {{"id": 5, "question": "Question 5?", "options": ["A", "B", "C", "D"], "correct_index": 1}},
    {{"id": 6, "question": "Question 6?", "options": ["A", "B", "C", "D"], "correct_index": 2}},
    {{"id": 7, "question": "Question 7?", "options": ["A", "B", "C", "D"], "correct_index": 3}},
    {{"id": 8, "question": "Question 8?", "options": ["A", "B", "C", "D"], "correct_index": 0}},
    {{"id": 9, "question": "Question 9?", "options": ["A", "B", "C", "D"], "correct_index": 1}},
    {{"id": 10, "question": "Question 10?", "options": ["A", "B", "C", "D"], "correct_index": 2}}
  ]
}}"#,
        training_content.title, topics_summary
    );

    let response = call_claude(&prompt).await?;
    parse_json_from_response(&response)
}

pub async fn answer_question(question: &str, context: &str) -> Result<String, String> {
    let prompt = format!(
        r#"You are a helpful training assistant. Based on this training content:

{}

Answer the following question concisely and helpfully:
{}

Provide a clear, educational answer."#,
        context, question
    );

    call_claude(&prompt).await
}

fn parse_json_from_response<T: serde::de::DeserializeOwned>(response: &str) -> Result<T, String> {
    let start = response.find('{');
    let end = response.rfind('}');

    match (start, end) {
        (Some(s), Some(e)) if s < e => {
            let json_str = &response[s..=e];
            serde_json::from_str(json_str)
                .map_err(|e| format!("Failed to parse JSON: {} - Response: {}", e, &json_str[..json_str.len().min(500)]))
        }
        _ => Err(format!("No valid JSON found in response: {}", &response[..response.len().min(500)])),
    }
}
