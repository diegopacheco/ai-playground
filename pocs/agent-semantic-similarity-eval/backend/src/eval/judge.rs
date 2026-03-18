use crate::agents::runner::AgentRunner;

pub struct JudgeEval;

impl JudgeEval {
    pub async fn score(question: &str, golden: &str, candidate: &str) -> Result<(f64, String), String> {
        let prompt = format!(
            "You are an expert LLM judge evaluating answer quality. \
            Given a question, a golden (reference) answer, and a candidate answer, \
            score how well the candidate captures the meaning and correctness of the golden answer. \
            Consider: factual accuracy, completeness, and semantic equivalence. \
            An answer can be worded completely differently but still be correct. \
            Return ONLY a single JSON object with this exact format, no other text: \
            {{\"score\": 0.XX, \"reasoning\": \"brief explanation\"}} \
            where score is between 0.0 and 1.0. \
            Question: \"{}\" \
            Golden Answer: \"{}\" \
            Candidate Answer: \"{}\"",
            question.replace("\"", "\\\""),
            golden.replace("\"", "\\\""),
            candidate.replace("\"", "\\\"")
        );

        let response = AgentRunner::call_llm(&prompt).await?;
        Self::parse_score(&response)
    }

    fn parse_score(response: &str) -> Result<(f64, String), String> {
        let json_start = response.find('{').ok_or("No JSON found in judge response")?;
        let json_end = response.rfind('}').ok_or("No JSON end found")? + 1;
        let json_str = &response[json_start..json_end];

        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse judge JSON: {} from: {}", e, json_str))?;

        let score = parsed["score"].as_f64()
            .ok_or_else(|| format!("No score field in judge response: {}", json_str))?;
        let reasoning = parsed["reasoning"].as_str()
            .unwrap_or("no reasoning provided")
            .to_string();

        Ok((score, reasoning))
    }
}
