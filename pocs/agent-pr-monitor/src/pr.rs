use std::fs;
use std::path::Path;
use std::process::Command;

pub fn parse_pr_url(url: &str) -> Result<(String, String, u64), String> {
    let url = url.trim().trim_end_matches('/');
    let parts: Vec<&str> = url.split('/').collect();
    if parts.len() < 7 || parts[5] != "pull" {
        return Err(format!("Invalid PR URL: {}", url));
    }
    let owner = parts[3].to_string();
    let repo = parts[4].to_string();
    let pr_number: u64 = parts[6].parse().map_err(|_| "Invalid PR number".to_string())?;
    Ok((owner, repo, pr_number))
}

pub fn clone_pr(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let clone_path = format!("/tmp/agent-pr/{}-{}-{}", owner, repo, pr_number);
    if Path::new(&clone_path).exists() {
        fs::remove_dir_all(&clone_path).map_err(|e| format!("Failed to clean: {}", e))?;
    }
    fs::create_dir_all(&clone_path).map_err(|e| format!("Failed to create dir: {}", e))?;

    let output = Command::new("gh")
        .args(["repo", "clone", &format!("{}/{}", owner, repo), &clone_path])
        .output()
        .map_err(|e| format!("Failed to clone: {}", e))?;
    if !output.status.success() {
        return Err(format!("Clone failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let output = Command::new("gh")
        .args(["pr", "checkout", &pr_number.to_string()])
        .current_dir(&clone_path)
        .output()
        .map_err(|e| format!("Failed to checkout PR: {}", e))?;
    if !output.status.success() {
        return Err(format!("PR checkout failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    Ok(clone_path)
}

pub fn git_pull(clone_path: &str) -> Result<String, String> {
    let output = Command::new("git")
        .args(["pull"])
        .current_dir(clone_path)
        .output()
        .map_err(|e| format!("git pull failed: {}", e))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() && (stdout.contains("CONFLICT") || stderr.contains("CONFLICT")) {
        return Err(format!("Merge conflict: {} {}", stdout, stderr));
    }
    if !output.status.success() {
        return Err(format!("git pull failed: {} {}", stdout, stderr));
    }
    Ok(stdout)
}

pub fn git_add_commit_push(clone_path: &str, message: &str) -> Result<String, String> {
    let output = Command::new("git")
        .args(["add", "-A"])
        .current_dir(clone_path)
        .output()
        .map_err(|e| format!("git add failed: {}", e))?;
    if !output.status.success() {
        return Err(format!("git add failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let output = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(clone_path)
        .output()
        .map_err(|e| format!("git commit failed: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !stderr.contains("nothing to commit") {
            return Err(format!("git commit failed: {}", stderr));
        }
    }

    let output = Command::new("git")
        .args(["push"])
        .current_dir(clone_path)
        .output()
        .map_err(|e| format!("git push failed: {}", e))?;
    if !output.status.success() {
        return Err(format!("git push failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn get_pr_title(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "pr", "view", &pr_number.to_string(),
            "--repo", &format!("{}/{}", owner, repo),
            "--json", "title", "-q", ".title",
        ])
        .output()
        .map_err(|e| format!("Failed to get PR title: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get PR title: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn get_pr_description(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "pr", "view", &pr_number.to_string(),
            "--repo", &format!("{}/{}", owner, repo),
            "--json", "body", "-q", ".body",
        ])
        .output()
        .map_err(|e| format!("Failed to get PR description: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get PR description: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn get_pr_branch(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "pr", "view", &pr_number.to_string(),
            "--repo", &format!("{}/{}", owner, repo),
            "--json", "headRefName", "-q", ".headRefName",
        ])
        .output()
        .map_err(|e| format!("Failed to get PR branch: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get PR branch: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn get_pr_base_branch(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "pr", "view", &pr_number.to_string(),
            "--repo", &format!("{}/{}", owner, repo),
            "--json", "baseRefName", "-q", ".baseRefName",
        ])
        .output()
        .map_err(|e| format!("Failed to get PR base branch: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get PR base branch: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn merge_base_branch(clone_path: &str, base_branch: &str) -> Result<String, String> {
    let output = Command::new("git")
        .args(["fetch", "origin", base_branch])
        .current_dir(clone_path)
        .output()
        .map_err(|e| format!("git fetch failed: {}", e))?;
    if !output.status.success() {
        return Err(format!("git fetch failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let output = Command::new("git")
        .args(["merge", &format!("origin/{}", base_branch), "--no-edit"])
        .current_dir(clone_path)
        .output()
        .map_err(|e| format!("git merge failed: {}", e))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() && (stdout.contains("CONFLICT") || stderr.contains("CONFLICT")) {
        return Err(format!("Merge conflict: {} {}", stdout, stderr));
    }
    if !output.status.success() {
        return Err(format!("git merge failed: {} {}", stdout, stderr));
    }
    Ok(stdout)
}

pub fn get_pr_changed_files(owner: &str, repo: &str, pr_number: u64) -> Result<Vec<String>, String> {
    let output = Command::new("gh")
        .args([
            "pr", "view", &pr_number.to_string(),
            "--repo", &format!("{}/{}", owner, repo),
            "--json", "files", "-q", ".files[].path",
        ])
        .output()
        .map_err(|e| format!("Failed to get PR changed files: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get PR changed files: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    Ok(files)
}

pub fn get_changed_dirs(clone_path: &str, changed_files: &[String]) -> Vec<String> {
    let mut dirs = std::collections::HashSet::new();
    for f in changed_files {
        let full = format!("{}/{}", clone_path, f);
        if let Some(parent) = Path::new(&full).parent() {
            dirs.insert(parent.to_string_lossy().to_string());
        }
    }
    dirs.into_iter().collect()
}

pub fn list_changed_source_files(clone_path: &str, changed_files: &[String]) -> Vec<String> {
    let dirs = get_changed_dirs(clone_path, changed_files);
    let mut files = Vec::new();
    for dir in &dirs {
        list_source_files_recursive(Path::new(dir), &mut files);
    }
    files
}

pub fn get_pr_diff(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "pr", "diff", &pr_number.to_string(),
            "--repo", &format!("{}/{}", owner, repo),
        ])
        .output()
        .map_err(|e| format!("Failed to get PR diff: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get PR diff: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn get_pr_comments(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "api",
            &format!("repos/{}/{}/pulls/{}/comments", owner, repo, pr_number),
        ])
        .output()
        .map_err(|e| format!("Failed to get PR comments: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get PR comments: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn get_pr_review_comments(owner: &str, repo: &str, pr_number: u64) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "api",
            &format!("repos/{}/{}/issues/{}/comments", owner, repo, pr_number),
        ])
        .output()
        .map_err(|e| format!("Failed to get review comments: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to get review comments: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn post_pr_comment(owner: &str, repo: &str, pr_number: u64, body: &str) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "pr", "comment", &pr_number.to_string(),
            "--repo", &format!("{}/{}", owner, repo),
            "--body", body,
        ])
        .output()
        .map_err(|e| format!("Failed to post comment: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to post comment: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn reply_to_review_comment(
    owner: &str, repo: &str, _pr_number: u64, comment_id: u64, body: &str,
) -> Result<String, String> {
    let output = Command::new("gh")
        .args([
            "api",
            &format!("repos/{}/{}/pulls/comments/{}/replies", owner, repo, comment_id),
            "-f", &format!("body={}", body),
        ])
        .output()
        .map_err(|e| format!("Failed to reply: {}", e))?;
    if !output.status.success() {
        return Err(format!("Failed to reply: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn count_files(clone_path: &str) -> usize {
    count_files_recursive(Path::new(clone_path))
}

fn count_files_recursive(path: &Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".git" {
                continue;
            }
            if entry_path.is_dir() {
                count += count_files_recursive(&entry_path);
            } else {
                count += 1;
            }
        }
    }
    count
}

pub fn read_file(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))
}

pub fn write_file(path: &str, content: &str) -> Result<(), String> {
    fs::write(path, content).map_err(|e| format!("Failed to write {}: {}", path, e))
}

fn is_source_file(name: &str) -> bool {
    let extensions = [
        ".rs", ".go", ".java", ".kt", ".scala",
        ".ts", ".tsx", ".js", ".jsx", ".mjs",
        ".py", ".rb", ".c", ".cpp", ".h", ".hpp",
        ".cs", ".swift", ".toml", ".yaml", ".yml",
        ".json", ".xml", ".html", ".css", ".scss",
        ".sql", ".sh", ".bash", ".zsh",
        ".mod", ".sum", ".lock", ".gradle",
    ];
    let lower = name.to_lowercase();
    extensions.iter().any(|ext| lower.ends_with(ext))
        || lower == "makefile"
        || lower == "dockerfile"
        || lower == "containerfile"
        || lower == "cargo.toml"
        || lower == "pom.xml"
        || lower == "build.gradle"
        || lower == "package.json"
}

fn list_source_files_recursive(path: &Path, files: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".git" || name == "target" || name == "node_modules"
                || name == "build" || name == "dist" || name == ".gradle"
            {
                continue;
            }
            if entry_path.is_dir() {
                list_source_files_recursive(&entry_path, files);
            } else if is_source_file(&name) {
                files.push(entry_path.to_string_lossy().to_string());
            }
        }
    }
}

pub fn get_conflicted_files(clone_path: &str) -> Vec<String> {
    let output = Command::new("git")
        .args(["diff", "--name-only", "--diff-filter=U"])
        .current_dir(clone_path)
        .output();
    match output {
        Ok(o) => {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect()
        }
        Err(_) => Vec::new(),
    }
}
