use crate::models::DimensionScores;
use serde::Deserialize;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

#[derive(Debug, Deserialize)]
struct ScoresResponse {
    quality: u8,
    stack_definitions: u8,
    clear_goals: u8,
    non_obvious_decisions: u8,
    security_operations: u8,
    overall_effectiveness: u8,
}

pub async fn calculate_scores(prompt: &str) -> DimensionScores {
    let analysis_prompt = format!(
        r#"Analyze the following prompt and score it on these 6 dimensions (1-5 scale each):

1. Quality: Overall prompt clarity, structure, grammar, and professionalism
2. Stack Definitions: How well technical stack, tools, and technologies are specified
3. Clear Goals: How clearly objectives, expected outcomes, and deliverables are defined
4. Non-obvious Decisions: Coverage of edge cases, constraints, trade-offs, and architectural choices
5. Security & Operations: Attention to security, authentication, deployment, monitoring, and operational concerns
6. Overall Effectiveness: Combined effectiveness as a prompt for AI coding assistants

Prompt to analyze:
---
{}
---

Respond ONLY with valid JSON in this exact format, nothing else:
{{"quality": N, "stack_definitions": N, "clear_goals": N, "non_obvious_decisions": N, "security_operations": N, "overall_effectiveness": N}}"#,
        prompt
    );

    let result = timeout(Duration::from_secs(60), async {
        Command::new("claude")
            .arg("-p")
            .arg(&analysis_prompt)
            .arg("--model")
            .arg("claude-opus-4-5-20251101")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
    })
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_scores_response(&stdout)
        }
        _ => default_scores(),
    }
}

fn parse_scores_response(output: &str) -> DimensionScores {
    let json_start = output.find('{');
    let json_end = output.rfind('}');

    if let (Some(start), Some(end)) = (json_start, json_end) {
        let json_str = &output[start..=end];
        if let Ok(response) = serde_json::from_str::<ScoresResponse>(json_str) {
            return DimensionScores {
                quality: response.quality.min(5).max(1),
                stack_definitions: response.stack_definitions.min(5).max(1),
                clear_goals: response.clear_goals.min(5).max(1),
                non_obvious_decisions: response.non_obvious_decisions.min(5).max(1),
                security_operations: response.security_operations.min(5).max(1),
                overall_effectiveness: response.overall_effectiveness.min(5).max(1),
            };
        }
    }

    default_scores()
}

fn default_scores() -> DimensionScores {
    DimensionScores {
        quality: 1,
        stack_definitions: 1,
        clear_goals: 1,
        non_obvious_decisions: 1,
        security_operations: 1,
        overall_effectiveness: 1,
    }
}
