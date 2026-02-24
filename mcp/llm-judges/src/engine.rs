use serde::{Deserialize, Serialize};
use crate::judges;

#[derive(Serialize, Deserialize, Clone)]
pub struct JudgeVerdict {
    pub name: String,
    pub verdict: String,
    pub confidence: String,
    pub reasoning: String,
}

#[derive(Serialize, Deserialize)]
pub struct JudgmentResult {
    pub verdict: String,
    pub score: String,
    pub judges: Vec<JudgeVerdict>,
    pub summary: String,
}

fn build_prompt(content: &str, criteria: &str) -> String {
    format!(
        "You are an impartial judge evaluating the following content.\n\n\
         CRITERIA: {}\n\n\
         CONTENT TO JUDGE:\n---\n{}\n---\n\n\
         Evaluate the content and respond in this exact format:\n\
         VERDICT: PASS or FAIL or UNCERTAIN\n\
         CONFIDENCE: high or medium or low\n\
         REASONING: Your explanation in 2-3 sentences.",
        criteria, content
    )
}

fn parse_verdict(raw: &str) -> (String, String, String) {
    let mut verdict = "UNCERTAIN".to_string();
    let mut confidence = "low".to_string();
    let mut reasoning = raw.to_string();

    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(v) = trimmed.strip_prefix("VERDICT:") {
            let v = v.trim().to_uppercase();
            if v.contains("PASS") {
                verdict = "PASS".to_string();
            } else if v.contains("FAIL") {
                verdict = "FAIL".to_string();
            } else {
                verdict = "UNCERTAIN".to_string();
            }
        } else if let Some(c) = trimmed.strip_prefix("CONFIDENCE:") {
            let c = c.trim().to_lowercase();
            if c.contains("high") {
                confidence = "high".to_string();
            } else if c.contains("medium") {
                confidence = "medium".to_string();
            } else {
                confidence = "low".to_string();
            }
        } else if let Some(r) = trimmed.strip_prefix("REASONING:") {
            reasoning = r.trim().to_string();
        }
    }

    (verdict, confidence, reasoning)
}

fn aggregate(verdicts: &[JudgeVerdict]) -> (String, String, String) {
    let total = verdicts.len();
    let pass_count = verdicts.iter().filter(|v| v.verdict == "PASS").count();
    let fail_count = verdicts.iter().filter(|v| v.verdict == "FAIL").count();

    let final_verdict = if pass_count > total / 2 {
        "PASS"
    } else if fail_count > total / 2 {
        "FAIL"
    } else {
        "SPLIT"
    };

    let score = format!("{}/{}", pass_count, total);
    let summary = format!(
        "{} out of {} judges ruled PASS. Majority verdict: {}.",
        pass_count, total, final_verdict
    );

    (final_verdict.to_string(), score, summary)
}

pub async fn run_judges(content: &str, criteria: Option<&str>, judge_names: &[&str]) -> JudgmentResult {
    let criteria = criteria.unwrap_or("Check for factual accuracy, logical consistency, and correctness.");
    let prompt = build_prompt(content, criteria);

    let mut handles = Vec::new();
    for &name in judge_names {
        let runner = judges::get_runner(name);
        let p = prompt.clone();
        let judge_name = name.to_string();
        handles.push(tokio::spawn(async move {
            match runner.run(&p).await {
                Ok(raw) => {
                    let (verdict, confidence, reasoning) = parse_verdict(&raw);
                    JudgeVerdict {
                        name: judge_name,
                        verdict,
                        confidence,
                        reasoning,
                    }
                }
                Err(e) => JudgeVerdict {
                    name: judge_name,
                    verdict: "ERROR".to_string(),
                    confidence: "low".to_string(),
                    reasoning: e,
                },
            }
        }));
    }

    let mut verdicts = Vec::new();
    for handle in handles {
        if let Ok(v) = handle.await {
            verdicts.push(v);
        }
    }

    let (final_verdict, score, summary) = aggregate(&verdicts);

    JudgmentResult {
        verdict: final_verdict,
        score,
        judges: verdicts,
        summary,
    }
}
