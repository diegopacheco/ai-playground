use crate::github::types::{FetchedCommit, GitHubCommitDetail, GitHubEvent};
use std::collections::HashSet;

pub async fn fetch_user_events(username: &str) -> Result<Vec<GitHubEvent>, reqwest::Error> {
    let url = format!(
        "https://api.github.com/users/{}/events?per_page=30",
        username
    );
    let client = reqwest::Client::new();
    client
        .get(&url)
        .header("User-Agent", "truth-detector")
        .send()
        .await?
        .json::<Vec<GitHubEvent>>()
        .await
}

pub async fn fetch_commit_detail(
    repo_full_name: &str,
    sha: &str,
) -> Result<GitHubCommitDetail, reqwest::Error> {
    let url = format!(
        "https://api.github.com/repos/{}/commits/{}",
        repo_full_name, sha
    );
    let client = reqwest::Client::new();
    client
        .get(&url)
        .header("User-Agent", "truth-detector")
        .send()
        .await?
        .json::<GitHubCommitDetail>()
        .await
}

pub async fn fetch_commits(username: &str) -> Result<Vec<FetchedCommit>, Box<dyn std::error::Error + Send + Sync>> {
    let events = fetch_user_events(username).await?;
    let mut seen = HashSet::new();
    let mut commit_refs = Vec::new();

    for event in &events {
        if event.r#type != "PushEvent" {
            continue;
        }
        if let Some(commits) = &event.payload.commits {
            for c in commits {
                if seen.contains(&c.sha) {
                    continue;
                }
                seen.insert(c.sha.clone());
                commit_refs.push((event.repo.name.clone(), c.sha.clone(), c.message.clone(), event.created_at.clone()));
                if commit_refs.len() >= 10 {
                    break;
                }
            }
        }
        if commit_refs.len() >= 10 {
            break;
        }
    }

    let mut results = Vec::new();
    for (repo_name, sha, message, date) in &commit_refs {
        match fetch_commit_detail(repo_name, sha).await {
            Ok(detail) => {
                let mut diff = String::new();
                if let Some(files) = &detail.files {
                    for f in files {
                        if let Some(patch) = &f.patch {
                            diff.push_str(patch);
                            diff.push('\n');
                        }
                        if diff.len() > 2000 {
                            break;
                        }
                    }
                }
                if diff.len() > 2000 {
                    diff.truncate(2000);
                }
                results.push(FetchedCommit {
                    sha: sha.clone(),
                    message: message.clone(),
                    repo_name: repo_name.clone(),
                    date: date.clone(),
                    diff,
                });
            }
            Err(_) => {
                results.push(FetchedCommit {
                    sha: sha.clone(),
                    message: message.clone(),
                    repo_name: repo_name.clone(),
                    date: date.clone(),
                    diff: String::new(),
                });
            }
        }
    }

    Ok(results)
}
