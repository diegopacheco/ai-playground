use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct GitHubEvent {
    pub r#type: String,
    pub repo: GitHubRepo,
    pub payload: GitHubPayload,
    pub created_at: String,
}

#[derive(Deserialize)]
pub struct GitHubRepo {
    pub name: String,
}

#[derive(Deserialize)]
pub struct GitHubPayload {
    pub commits: Option<Vec<GitHubCommitRef>>,
    pub head: Option<String>,
}

#[derive(Deserialize)]
pub struct GitHubCommitRef {
    pub sha: String,
    pub message: String,
}

#[derive(Deserialize)]
pub struct GitHubCommitDetail {
    pub sha: String,
    pub commit: GitHubCommitInfo,
    pub files: Option<Vec<GitHubFile>>,
}

#[derive(Deserialize)]
pub struct GitHubCommitInfo {
    pub message: String,
    pub author: GitHubAuthor,
}

#[derive(Deserialize)]
pub struct GitHubAuthor {
    pub name: String,
    pub date: String,
}

#[derive(Deserialize)]
pub struct GitHubFile {
    pub filename: String,
    pub patch: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FetchedCommit {
    pub sha: String,
    pub message: String,
    pub repo_name: String,
    pub date: String,
    pub diff: String,
}
