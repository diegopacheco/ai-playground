use crate::models::types::{LlmCommitResult, WeeklyScore};
use chrono::{Datelike, Local};

pub fn calculate_scores(results: &[LlmCommitResult]) -> (i32, f64, i32, i32, i32) {
    let total_score: i32 = results.iter().map(|r| r.score).sum();
    let avg_score = if results.is_empty() {
        0.0
    } else {
        total_score as f64 / results.len() as f64
    };
    let deep_count = results
        .iter()
        .filter(|r| r.classification == "DEEP")
        .count() as i32;
    let decent_count = results
        .iter()
        .filter(|r| r.classification == "DECENT")
        .count() as i32;
    let shallow_count = results
        .iter()
        .filter(|r| r.classification == "SHALLOW")
        .count() as i32;
    (total_score, avg_score, deep_count, decent_count, shallow_count)
}

pub fn build_weekly_score(github_user: &str, results: &[LlmCommitResult]) -> WeeklyScore {
    let today = Local::now().date_naive();
    let days_since_monday = today.weekday().num_days_from_monday();
    let monday = today - chrono::Duration::days(days_since_monday as i64);
    let sunday = monday + chrono::Duration::days(6);

    let (total_score, avg_score, deep_count, decent_count, shallow_count) =
        calculate_scores(results);

    WeeklyScore {
        id: uuid::Uuid::new_v4().to_string(),
        github_user: github_user.to_string(),
        week_start: monday.format("%Y-%m-%d").to_string(),
        week_end: sunday.format("%Y-%m-%d").to_string(),
        total_score,
        avg_score,
        num_commits: results.len() as i32,
        deep_count,
        decent_count,
        shallow_count,
    }
}
