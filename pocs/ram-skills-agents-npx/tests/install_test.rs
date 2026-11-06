mod support;

use ram::catalog::{Item, scan};
use ram::install::{Installer, install};
use ram::kind::{Kind, Target};
use std::fs;

fn item(kind: Kind, name: &str) -> Item {
    scan(&support::fixture_repo()).unwrap().into_iter().find(|i| i.kind == kind && i.name == name).unwrap()
}

#[tokio::test]
async fn skill_lands_in_claude_skills_with_its_files_but_without_dependencies() {
    let home = support::temp_dir("home");
    let dest = install(&item(Kind::Skill, "sql-migration"), Target::Claude, &home).await.unwrap();
    let skill = home.join(".claude/skills/sql-migration");
    assert_eq!(dest, skill.to_string_lossy());
    assert!(skill.join("SKILL.md").is_file());
    assert!(skill.join("references/pg.md").is_file());
    let bash = home.join(".claude/skills/bash-skill");
    install(&item(Kind::Skill, "bash-skill"), Target::Claude, &home).await.unwrap();
    assert!(!bash.join("node_modules").exists());
}

#[tokio::test]
async fn reinstall_replaces_the_old_skill_so_removed_files_do_not_linger() {
    let home = support::temp_dir("home");
    support::write(&home, ".codex/skills/bash-skill/stale.md", "old");
    install(&item(Kind::Skill, "bash-skill"), Target::Codex, &home).await.unwrap();
    assert!(!home.join(".codex/skills/bash-skill/stale.md").exists());
    assert!(home.join(".codex/skills/bash-skill/SKILL.md").is_file());
}

#[tokio::test]
async fn commands_go_to_claude_commands_and_codex_prompts_under_their_slash_name() {
    let home = support::temp_dir("home");
    let installer = Installer { home: home.clone(), marketplace_source: String::new() };
    let outcomes = installer.install_all(vec![item(Kind::Command, "fact-check")], &[Target::Claude, Target::Codex]).await;
    assert!(outcomes.iter().all(|o| o.result.is_ok()));
    let claude = fs::read_to_string(home.join(".claude/commands/fact-check.md")).unwrap();
    assert!(claude.starts_with("# /fact-check"));
    assert!(home.join(".codex/prompts/fact-check.md").is_file());
}

#[tokio::test]
async fn agents_install_for_claude_and_are_skipped_for_codex() {
    let home = support::temp_dir("home");
    let installer = Installer { home: home.clone(), marketplace_source: String::new() };
    let outcomes = installer.install_all(vec![item(Kind::Agent, "code-reviewer")], &[Target::Claude, Target::Codex]).await;
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].target, Target::Claude);
    assert!(home.join(".claude/agents/code-reviewer.md").is_file());
    assert!(!home.join(".codex").exists());
}

#[tokio::test]
async fn asking_codex_for_an_agent_directly_fails_loud() {
    let home = support::temp_dir("home");
    let error = install(&item(Kind::Agent, "code-reviewer"), Target::Codex, &home).await.unwrap_err();
    assert!(error.to_string().contains("codex does not support agents"));
}

#[tokio::test]
async fn results_come_back_in_the_order_the_user_picked() {
    let home = support::temp_dir("home");
    let installer = Installer { home: home.clone(), marketplace_source: String::new() };
    let picked = vec![item(Kind::Command, "do"), item(Kind::Skill, "bash-skill"), item(Kind::Skill, "fact-checker")];
    let outcomes = installer.install_all(picked, &[Target::Claude]).await;
    let names: Vec<&str> = outcomes.iter().map(|o| o.item.name.as_str()).collect();
    assert_eq!(names, vec!["do", "bash-skill", "fact-checker"]);
}
