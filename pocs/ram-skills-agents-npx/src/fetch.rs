use crate::error::{Result, fail};
use std::path::{Path, PathBuf};
use tokio::process::Command;

pub const DEFAULT_REPO: &str = "https://github.com/diegopacheco/agent-skills";

pub trait Source {
    fn label(&self) -> String;
    fn fetch(&self) -> impl Future<Output = Result<PathBuf>> + Send;
}

pub struct GitSource {
    pub url: String,
    pub cache: PathBuf,
}

impl GitSource {
    pub fn new(url: &str, home: &Path) -> Self {
        let slug: String = url
            .trim_end_matches(".git")
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        GitSource {
            url: url.to_string(),
            cache: home.join(".cache").join("ram").join(slug.trim_matches('-')),
        }
    }

    pub fn marketplace_source(&self) -> String {
        self.url
            .strip_prefix("https://github.com/")
            .map(|rest| rest.trim_end_matches(".git").to_string())
            .unwrap_or_else(|| self.url.clone())
    }
}

impl Source for GitSource {
    fn label(&self) -> String {
        self.url.clone()
    }

    async fn fetch(&self) -> Result<PathBuf> {
        let dir = self.cache.to_string_lossy().to_string();
        if self.cache.join(".git").is_dir() {
            git(&["-C", &dir, "fetch", "--quiet", "--depth", "1", "origin", "HEAD"]).await?;
            git(&["-C", &dir, "reset", "--quiet", "--hard", "FETCH_HEAD"]).await?;
        } else {
            if let Some(parent) = self.cache.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            if self.cache.exists() {
                tokio::fs::remove_dir_all(&self.cache).await?;
            }
            git(&["clone", "--quiet", "--depth", "1", &self.url, &dir]).await?;
        }
        Ok(self.cache.clone())
    }
}

pub async fn revision(dir: &Path) -> Result<String> {
    git(&["-C", &dir.to_string_lossy(), "rev-parse", "--short", "HEAD"]).await
}

async fn git(args: &[&str]) -> Result<String> {
    let output = Command::new("git").args(args).output().await?;
    if !output.status.success() {
        return fail(format!("git {} failed: {}", args.join(" "), String::from_utf8_lossy(&output.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
