use std::path::PathBuf;
use std::process::Command;
use tokio::fs;

pub async fn create_workspace(project_name: &str) -> Result<PathBuf, String> {
    let base_path = std::env::current_dir()
        .map_err(|e| e.to_string())?
        .join("workspaces")
        .join(project_name);
    fs::create_dir_all(&base_path).await.map_err(|e| e.to_string())?;
    let base_repo = base_path.join("base");
    if !base_repo.exists() {
        fs::create_dir_all(&base_repo).await.map_err(|e| e.to_string())?;
        let output = Command::new("git")
            .args(["init"])
            .current_dir(&base_repo)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }
        let gitignore_path = base_repo.join(".gitignore");
        fs::write(&gitignore_path, "target/\nnode_modules/\n").await.map_err(|e| e.to_string())?;
        Command::new("git")
            .args(["add", "."])
            .current_dir(&base_repo)
            .output()
            .map_err(|e| e.to_string())?;
        Command::new("git")
            .args(["commit", "-m", "Initial commit"])
            .current_dir(&base_repo)
            .output()
            .map_err(|e| e.to_string())?;
    }
    Ok(base_path)
}

pub async fn create_agent_worktree(base_path: &PathBuf, agent_name: &str) -> Result<PathBuf, String> {
    let base_repo = base_path.join("base");
    let worktree_path = base_path.join(agent_name);
    let worktree_abs = worktree_path.canonicalize().unwrap_or_else(|_| worktree_path.clone());
    if worktree_path.exists() {
        fs::remove_dir_all(&worktree_path).await.map_err(|e| e.to_string())?;
        Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&base_repo)
            .output()
            .map_err(|e| e.to_string())?;
    }
    let branch_name = format!("agent-{}", agent_name);
    Command::new("git")
        .args(["branch", "-D", &branch_name])
        .current_dir(&base_repo)
        .output()
        .ok();
    let worktree_str = if worktree_abs.is_absolute() {
        worktree_abs.to_string_lossy().to_string()
    } else {
        base_path.join(agent_name).to_string_lossy().to_string()
    };
    let output = Command::new("git")
        .args(["worktree", "add", "-b", &branch_name, &worktree_str])
        .current_dir(&base_repo)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(worktree_path)
}
