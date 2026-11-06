use crate::catalog::Item;
use crate::error::{Result, fail};
use crate::kind::{Kind, Target};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::process::Command;

const SKIPPED: [&str; 2] = [".git", "node_modules"];

#[derive(Debug)]
pub struct Outcome {
    pub item: Item,
    pub target: Target,
    pub result: Result<String>,
}

pub struct Installer {
    pub home: PathBuf,
    pub marketplace_source: String,
}

impl Installer {
    pub async fn install_all(&self, items: Vec<Item>, targets: &[Target]) -> Vec<Outcome> {
        let jobs: Vec<(Item, Target)> = items
            .iter()
            .flat_map(|item| targets.iter().filter(|t| t.supports(item.kind)).map(move |t| (item.clone(), *t)))
            .collect();
        if jobs.iter().any(|(item, _)| item.kind == Kind::Plugin)
            && let Err(error) = add_marketplace(&self.marketplace_source).await
        {
            return jobs
                .into_iter()
                .map(|(item, target)| Outcome { item, target, result: Err(error.to_string().into()) })
                .collect();
        }
        let mut set = tokio::task::JoinSet::new();
        for (index, (item, target)) in jobs.into_iter().enumerate() {
            let home = self.home.clone();
            set.spawn(async move {
                let result = install(&item, target, &home).await;
                (index, Outcome { item, target, result })
            });
        }
        let mut outcomes = Vec::new();
        while let Some(joined) = set.join_next().await {
            if let Ok(pair) = joined {
                outcomes.push(pair);
            }
        }
        outcomes.sort_by_key(|(index, _)| *index);
        outcomes.into_iter().map(|(_, outcome)| outcome).collect()
    }
}

pub async fn install(item: &Item, target: Target, home: &Path) -> Result<String> {
    if !target.supports(item.kind) {
        return fail(format!("{} does not support {}s", target.label(), item.kind.label()));
    }
    let dir = target.dir(home, item.kind);
    match item.kind {
        Kind::Skill => install_skill(item, &dir.join(&item.name)).await,
        Kind::Command | Kind::Agent => install_file(item, &dir.join(format!("{}.md", item.name))).await,
        Kind::Plugin => install_plugin(item).await,
    }
}

async fn install_skill(item: &Item, dest: &Path) -> Result<String> {
    if fs::try_exists(dest).await? {
        fs::remove_dir_all(dest).await?;
    }
    copy_dir(&item.source, dest).await?;
    if fs::try_exists(dest.join("package.json")).await? {
        run("npm", &["install", "--silent", "--no-audit", "--no-fund"], Some(dest)).await?;
    }
    Ok(dest.to_string_lossy().to_string())
}

async fn install_file(item: &Item, dest: &Path) -> Result<String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::copy(&item.source, dest).await?;
    Ok(dest.to_string_lossy().to_string())
}

async fn install_plugin(item: &Item) -> Result<String> {
    let id = match &item.marketplace {
        Some(marketplace) => format!("{}@{}", item.name, marketplace),
        None => item.name.clone(),
    };
    run("claude", &["plugin", "install", &id], None).await?;
    Ok(format!("claude plugin {id}"))
}

async fn add_marketplace(source: &str) -> Result<()> {
    let listed = run("claude", &["plugin", "marketplace", "list"], None).await?;
    if !listed.contains(source) {
        run("claude", &["plugin", "marketplace", "add", source], None).await?;
    }
    Ok(())
}

async fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    let mut stack = vec![(from.to_path_buf(), to.to_path_buf())];
    while let Some((src, dst)) = stack.pop() {
        fs::create_dir_all(&dst).await?;
        let mut entries = fs::read_dir(&src).await?;
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name();
            if SKIPPED.iter().any(|s| name == *s) {
                continue;
            }
            let path = entry.path();
            if entry.file_type().await?.is_dir() {
                stack.push((path, dst.join(&name)));
            } else {
                fs::copy(&path, dst.join(&name)).await?;
            }
        }
    }
    Ok(())
}

async fn run(program: &str, args: &[&str], cwd: Option<&Path>) -> Result<String> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }
    let output = command.output().await.map_err(|e| format!("{program}: {e}"))?;
    if !output.status.success() {
        return fail(format!("{program} {} failed: {}", args.join(" "), String::from_utf8_lossy(&output.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
