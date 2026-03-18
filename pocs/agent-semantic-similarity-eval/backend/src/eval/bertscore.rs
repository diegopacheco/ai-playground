use crate::agents::runner::AgentRunner;

pub struct BertScoreEval;

impl BertScoreEval {
    pub async fn score(golden: &str, candidate: &str) -> Result<f64, String> {
        let prompt = format!(
            "You are a semantic similarity evaluator approximating BERTScore. \
            Compare the candidate text against the reference text for semantic overlap. \
            Consider precision (how much of candidate is in reference), \
            recall (how much of reference is in candidate), \
            and F1 (harmonic mean of both). \
            Return ONLY a single JSON object with this exact format, no other text: \
            {{\"precision\": 0.XX, \"recall\": 0.XX, \"f1\": 0.XX}} \
            Reference: \"{}\" \
            Candidate: \"{}\"",
            golden.replace("\"", "\\\""),
            candidate.replace("\"", "\\\"")
        );

        let response = AgentRunner::call_llm(&prompt).await?;
        Self::parse_score(&response)
    }

    fn parse_score(response: &str) -> Result<f64, String> {
        let json_start = response.find('{').ok_or("No JSON found in BERTScore response")?;
        let json_end = response.rfind('}').ok_or("No JSON end found")? + 1;
        let json_str = &response[json_start..json_end];

        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse BERTScore JSON: {} from: {}", e, json_str))?;

        parsed["f1"].as_f64()
            .ok_or_else(|| format!("No f1 field in BERTScore response: {}", json_str))
    }
}
