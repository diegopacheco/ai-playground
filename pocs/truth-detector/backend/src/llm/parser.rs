use crate::github::types::FetchedCommit;
use crate::llm::prompt::group_by_day;
use crate::models::types::LlmCommitResult;
use serde::Deserialize;

#[derive(Deserialize)]
struct DayResult {
    day_index: usize,
    classification: String,
    summary: String,
    score: i32,
}

#[derive(Deserialize)]
struct DayBatchResponse {
    results: Vec<DayResult>,
}

pub fn parse_batch_response(raw: &str, commits: &[FetchedCommit]) -> Vec<LlmCommitResult> {
    let groups = group_by_day(commits);

    if let Some(parsed) = try_parse(raw) {
        let mut commit_results = Vec::new();
        let mut day_map = std::collections::HashMap::new();

        for dr in &parsed.results {
            if dr.day_index >= 1 && dr.day_index <= groups.len() {
                day_map.insert(dr.day_index, dr);
            }
        }

        for (gi, group) in groups.iter().enumerate() {
            let day_idx = gi + 1;
            if let Some(dr) = day_map.get(&day_idx) {
                for &ci in &group.commits {
                    commit_results.push(LlmCommitResult {
                        index: ci + 1,
                        classification: dr.classification.clone(),
                        summary: dr.summary.clone(),
                        score: dr.score,
                    });
                }
            } else {
                for &ci in &group.commits {
                    commit_results.push(fallback_entry(ci + 1, &commits[ci]));
                }
            }
        }

        commit_results.sort_by_key(|r| r.index);
        commit_results
    } else {
        commits
            .iter()
            .enumerate()
            .map(|(i, c)| fallback_entry(i + 1, c))
            .collect()
    }
}

fn try_parse(raw: &str) -> Option<DayBatchResponse> {
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
