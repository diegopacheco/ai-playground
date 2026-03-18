use serde::{Deserialize, Serialize};
use crate::eval::cosine::CosineEval;
use crate::eval::bertscore::BertScoreEval;
use crate::eval::judge::JudgeEval;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EvalResult {
    pub question: String,
    pub golden_answer: String,
    pub candidate_answer: String,
    pub cosine_score: f64,
    pub bert_score: f64,
    pub judge_score: f64,
    pub judge_reasoning: String,
    pub final_score: f64,
    pub verdict: String,
}

pub struct Scorer;

impl Scorer {
    pub async fn evaluate(question: &str, golden: &str, candidate: &str) -> Result<EvalResult, String> {
        let cosine_score = CosineEval::score(golden, candidate);

        let bert_score = BertScoreEval::score(golden, candidate).await?;

        let (judge_score, judge_reasoning) = JudgeEval::score(question, golden, candidate).await?;

        let final_score = (cosine_score * 0.25) + (bert_score * 0.35) + (judge_score * 0.40);

        let verdict = if final_score >= 0.8 {
            "PASS - Strong semantic match".to_string()
        } else if final_score >= 0.6 {
            "MARGINAL - Partial match, review recommended".to_string()
        } else {
            "FAIL - Weak semantic match".to_string()
        };

        Ok(EvalResult {
            question: question.to_string(),
            golden_answer: golden.to_string(),
            candidate_answer: candidate.to_string(),
            cosine_score,
            bert_score,
            judge_score,
            judge_reasoning,
            final_score,
            verdict,
        })
    }
}
