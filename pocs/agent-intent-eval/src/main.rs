use std::process::Command;
use std::time::Duration;
use tokio::time::timeout;

struct EvalResult {
    paraphrase: String,
    response: String,
}

struct SimilarityScore {
    pair: (usize, usize),
    score: f64,
}

fn generate_paraphrases(base_prompt: &str) -> Vec<String> {
    vec![
        base_prompt.to_string(),
        format!("Can you tell me: {}", base_prompt),
        format!("I'd like to know: {}", base_prompt),
        format!("Please explain: {}", base_prompt),
        format!("In your own words, {}", base_prompt.to_lowercase()),
        format!("Help me understand: {}", base_prompt),
        format!("Could you clarify: {}", base_prompt),
        format!("What is your take on: {}", base_prompt.to_lowercase()),
        format!("Describe for me: {}", base_prompt.to_lowercase()),
        format!("Give me a detailed answer: {}", base_prompt.to_lowercase()),
    ]
}

async fn call_llm(prompt: &str) -> Result<String, String> {
    let result = timeout(Duration::from_secs(120), async {
        let output = Command::new("claude")
            .args(&["-p", prompt, "--model", "sonnet", "--dangerously-skip-permissions"])
            .output()
            .map_err(|e| format!("Failed to execute claude CLI: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("claude CLI error: {}", stderr))
        }
    })
    .await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err("LLM call timed out after 120s".to_string()),
    }
}

async fn judge_similarity(response_a: &str, response_b: &str) -> f64 {
    let judge_prompt = format!(
        "You are a semantic similarity judge. Compare these two responses and rate their semantic similarity from 0.0 to 1.0 where 1.0 means identical meaning and 0.0 means completely different. Only respond with a single decimal number, nothing else.\n\nResponse A:\n{}\n\nResponse B:\n{}",
        response_a, response_b
    );

    match call_llm(&judge_prompt).await {
        Ok(score_str) => {
            score_str
                .trim()
                .parse::<f64>()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0)
        }
        Err(_) => 0.0,
    }
}

fn print_report(results: &[EvalResult], scores: &[SimilarityScore], avg_score: f64) {
    println!("\n{}", "=".repeat(80));
    println!("INTENT PRESERVATION EVALUATION REPORT");
    println!("{}", "=".repeat(80));

    for (i, result) in results.iter().enumerate() {
        println!("\n--- Paraphrase {} ---", i + 1);
        println!("Prompt: {}", result.paraphrase);
        let truncated = &result.response[..result.response.len().min(200)];
        println!("Response (first 200 chars): {}", truncated);
    }

    println!("\n{}", "-".repeat(80));
    println!("PAIRWISE SIMILARITY SCORES");
    println!("{}", "-".repeat(80));

    for s in scores {
        println!("  Pair ({}, {}): {:.2}", s.pair.0 + 1, s.pair.1 + 1, s.score);
    }

    println!("\n{}", "-".repeat(80));
    println!("OVERALL ROBUSTNESS SCORE: {:.2}", avg_score);
    println!("{}", "-".repeat(80));

    if avg_score >= 0.8 {
        println!("VERDICT: HIGH robustness - LLM gives semantically consistent answers across phrasings");
    } else if avg_score >= 0.5 {
        println!("VERDICT: MODERATE robustness - some variation in answers across phrasings");
    } else {
        println!("VERDICT: LOW robustness - LLM answers vary significantly with phrasing changes");
    }

    println!("{}", "=".repeat(80));
}

#[tokio::main]
async fn main() {
    let base_prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the capital of France?".to_string());

    println!("Intent Preservation Eval");
    println!("Base prompt: {}", base_prompt);
    println!("Generating 10 paraphrases and querying LLM...\n");

    let paraphrases = generate_paraphrases(&base_prompt);
    let mut results: Vec<EvalResult> = Vec::new();

    for (i, p) in paraphrases.iter().enumerate() {
        println!("[{}/10] Querying: {}", i + 1, p);
        match call_llm(p).await {
            Ok(response) => {
                println!("[{}/10] Got response ({} chars)", i + 1, response.len());
                results.push(EvalResult {
                    paraphrase: p.clone(),
                    response,
                });
            }
            Err(e) => {
                println!("[{}/10] Error: {}", i + 1, e);
                results.push(EvalResult {
                    paraphrase: p.clone(),
                    response: format!("ERROR: {}", e),
                });
            }
        }
    }

    println!("\nJudging pairwise semantic similarity...");

    let mut scores: Vec<SimilarityScore> = Vec::new();
    let baseline = &results[0].response;

    for i in 1..results.len() {
        println!("  Comparing pair (1, {})...", i + 1);
        let score = judge_similarity(baseline, &results[i].response).await;
        scores.push(SimilarityScore {
            pair: (0, i),
            score,
        });
    }

    let avg_score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64
    };

    print_report(&results, &scores, avg_score);
}
