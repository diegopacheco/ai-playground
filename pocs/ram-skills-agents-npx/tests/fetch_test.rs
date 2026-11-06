mod support;

use ram::fetch::{GitSource, Source, revision};
use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .args(["-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"])
        .args(args)
        .current_dir(dir)
        .status()
        .unwrap();
    assert!(status.success());
}

#[tokio::test]
async fn every_run_fetches_the_latest_commit_of_the_repo() {
    let upstream = support::fixture_repo();
    git(&upstream, &["init", "-q"]);
    git(&upstream, &["add", "."]);
    git(&upstream, &["commit", "-qm", "one"]);
    let home = support::temp_dir("home");
    let source = GitSource::new(&format!("file://{}", upstream.display()), &home);
    let first = source.fetch().await.unwrap();
    assert!(first.join("bash-skill/SKILL.md").is_file());
    let first_rev = revision(&first).await.unwrap();
    support::write(&upstream, "new-skill/SKILL.md", "---\nname: new-skill\n---\n");
    git(&upstream, &["add", "."]);
    git(&upstream, &["commit", "-qm", "two"]);
    let second = source.fetch().await.unwrap();
    assert_eq!(first, second);
    assert_ne!(revision(&second).await.unwrap(), first_rev);
    assert!(second.join("new-skill/SKILL.md").is_file());
}

#[test]
fn github_urls_become_owner_repo_marketplace_sources() {
    let source = GitSource::new("https://github.com/diegopacheco/agent-skills.git", Path::new("/h"));
    assert_eq!(source.marketplace_source(), "diegopacheco/agent-skills");
    assert!(source.cache.starts_with("/h/.cache/ram"));
}
