use crate::engine::scoring;
use crate::github::cache;
use crate::llm::{parser, prompt, runner};
use crate::models::types::{AppState, CommitResult, SseCommitAnalyzed, SseCommitInfo, SseEvent};
use crate::persistence::db;
use crate::sse::broadcaster;

pub async fn run_analysis(app_state: AppState, analysis_id: String, github_user: String, cli: String, model: String) {
    let sender = {
        let channels = &app_state.channels;
        let map = channels.lock().await;
        map.get(&analysis_id).cloned()
    };

    let result = run_inner(&app_state, &analysis_id, &github_user, &sender, &cli, &model).await;

    if let Err(e) = result {
        {
            let conn = app_state.db.lock().await;
            let _ = db::update_analysis_failed(&conn, &analysis_id);
        }
        if let Some(tx) = &sender {
            broadcaster::send_event(
                tx,
                SseEvent::Error {
                    message: e.to_string(),
                },
            );
        }
    }

    broadcaster::remove_channel(&app_state.channels, &analysis_id).await;
}

async fn run_inner(
    app_state: &AppState,
    analysis_id: &str,
    github_user: &str,
    sender: &Option<tokio::sync::broadcast::Sender<String>>,
    cli: &str,
    model: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (commits, cached) = cache::get_or_fetch(&app_state.db, github_user).await?;

    if let Some(tx) = sender {
        broadcaster::send_event(
            tx,
            SseEvent::AnalysisStart {
                github_user: github_user.to_string(),
                commit_count: commits.len(),
                cached,
            },
        );
    }

    if let Some(tx) = sender {
        let infos: Vec<SseCommitInfo> = commits
            .iter()
            .map(|c| SseCommitInfo {
                sha: c.sha.clone(),
                message: c.message.clone(),
                repo: c.repo_name.clone(),
            })
            .collect();
        broadcaster::send_event(tx, SseEvent::CommitsFetched { commits: infos });
    }

    if let Some(tx) = sender {
        broadcaster::send_event(tx, SseEvent::LlmThinking);
    }

    let batch_prompt = prompt::build_batch_prompt(&commits);
    let raw_output = runner::run_llm(&batch_prompt, cli, model).await?;
    let llm_results = parser::parse_batch_response(&raw_output, &commits);

    let commit_results: Vec<CommitResult> = llm_results
        .iter()
        .map(|lr| {
            let idx = lr.index - 1;
            let commit = &commits[idx.min(commits.len() - 1)];
            CommitResult {
                id: uuid::Uuid::new_v4().to_string(),
                analysis_id: analysis_id.to_string(),
                repo_name: commit.repo_name.clone(),
                commit_sha: commit.sha.clone(),
                commit_message: commit.message.clone(),
                commit_date: commit.date.clone(),
                classification: lr.classification.clone(),
                summary: lr.summary.clone(),
                score: lr.score,
                fallback: lr.classification == "DECENT" && lr.score == 5 && lr.summary == commit.message[..lr.summary.len().min(commit.message.len())],
                raw_llm_output: Some(raw_output.clone()),
            }
        })
        .collect();

    {
        let conn = app_state.db.lock().await;
        db::insert_commit_results(&conn, &commit_results)?;
    }

    if let Some(tx) = sender {
        let analyzed: Vec<SseCommitAnalyzed> = commit_results
            .iter()
            .enumerate()
            .map(|(i, cr)| SseCommitAnalyzed {
                index: i + 1,
                sha: cr.commit_sha.clone(),
                classification: cr.classification.clone(),
                summary: cr.summary.clone(),
                score: cr.score,
                fallback: cr.fallback,
            })
            .collect();
        broadcaster::send_event(tx, SseEvent::AllCommitsAnalyzed { results: analyzed });
    }

    let (total_score, avg_score, deep_count, decent_count, shallow_count) =
        scoring::calculate_scores(&llm_results);

    {
        let conn = app_state.db.lock().await;
        db::update_analysis_complete(&conn, analysis_id, total_score, avg_score)?;
    }

    let weekly = scoring::build_weekly_score(github_user, &llm_results);
    {
        let conn = app_state.db.lock().await;
        db::upsert_weekly_score(&conn, &weekly)?;
    }

    if let Some(tx) = sender {
        broadcaster::send_event(
            tx,
            SseEvent::AnalysisComplete {
                total_score,
                avg_score,
                deep: deep_count,
                decent: decent_count,
                shallow: shallow_count,
            },
        );
    }

    Ok(())
}
