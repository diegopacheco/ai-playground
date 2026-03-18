use axum::Json;
use serde::{Deserialize, Serialize};
use crate::eval::scorer::{Scorer, EvalResult};

#[derive(Debug, Deserialize)]
pub struct EvalRequest {
    pub question: String,
    pub golden_answer: String,
    pub candidate_answer: String,
}

#[derive(Debug, Deserialize)]
pub struct BatchEvalRequest {
    pub items: Vec<EvalRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchEvalResponse {
    pub results: Vec<EvalResult>,
    pub average_score: f64,
    pub pass_count: usize,
    pub fail_count: usize,
    pub marginal_count: usize,
}

pub async fn evaluate(Json(req): Json<EvalRequest>) -> Json<EvalResult> {
    let result = Scorer::evaluate(&req.question, &req.golden_answer, &req.candidate_answer)
        .await
        .unwrap_or_else(|e| EvalResult {
            question: req.question,
            golden_answer: req.golden_answer,
            candidate_answer: req.candidate_answer,
            cosine_score: 0.0,
            bert_score: 0.0,
            judge_score: 0.0,
            judge_reasoning: format!("Error: {}", e),
            final_score: 0.0,
            verdict: "ERROR".to_string(),
        });
    Json(result)
}

pub async fn batch_evaluate(Json(req): Json<BatchEvalRequest>) -> Json<BatchEvalResponse> {
    let mut results = Vec::new();
    let mut pass_count = 0;
    let mut fail_count = 0;
    let mut marginal_count = 0;

    for item in &req.items {
        let result = Scorer::evaluate(&item.question, &item.golden_answer, &item.candidate_answer)
            .await
            .unwrap_or_else(|e| EvalResult {
                question: item.question.clone(),
                golden_answer: item.golden_answer.clone(),
                candidate_answer: item.candidate_answer.clone(),
                cosine_score: 0.0,
                bert_score: 0.0,
                judge_score: 0.0,
                judge_reasoning: format!("Error: {}", e),
                final_score: 0.0,
                verdict: "ERROR".to_string(),
            });

        if result.verdict.starts_with("PASS") {
            pass_count += 1;
        } else if result.verdict.starts_with("FAIL") {
            fail_count += 1;
        } else if result.verdict.starts_with("MARGINAL") {
            marginal_count += 1;
        }

        results.push(result);
    }

    let average_score = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.final_score).sum::<f64>() / results.len() as f64
    };

    Json(BatchEvalResponse {
        results,
        average_score,
        pass_count,
        fail_count,
        marginal_count,
    })
}
