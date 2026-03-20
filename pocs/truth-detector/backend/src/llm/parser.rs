use crate::github::types::FetchedCommit;
use crate::models::types::{LlmBatchResponse, LlmCommitResult};

pub fn parse_batch_response(raw: &str, commits: &[FetchedCommit]) -> Vec<LlmCommitResult> {
    if let Some(parsed) = try_parse(raw) {
        let mut results = Vec::new();
        let mut found = std::collections::HashSet::new();
        for r in parsed.results {
            if r.index >= 1 && r.index <= commits.len() {
                found.insert(r.index);
                results.push(r);
            }
        }
        for i in 1..=commits.len() {
            if !found.contains(&i) {
                results.push(fallback_entry(i, &commits[i - 1]));
            }
        }
        results.sort_by_key(|r| r.index);
        results
    } else {
        commits
            .iter()
            .enumerate()
            .map(|(i, c)| fallback_entry(i + 1, c))
            .collect()
    }
}

fn try_parse(raw: &str) -> Option<LlmBatchResponse> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end < start {
        return None;
    }
    let json_str = &raw[start..=end];
    serde_json::from_str(json_str).ok()
}

fn fallback_entry(index: usize, commit: &FetchedCommit) -> LlmCommitResult {
    let summary = if commit.message.len() > 80 {
        commit.message[..80].to_string()
    } else {
        commit.message.clone()
    };
    LlmCommitResult {
        index,
        classification: "DECENT".to_string(),
        summary,
        score: 5,
    }
}
