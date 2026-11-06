mod support;

use ram::catalog::{Item, scan};
use ram::detail::load;
use ram::kind::{Kind, Target};
use std::path::Path;

fn item(kind: Kind, name: &str) -> Item {
    scan(&support::fixture_repo()).unwrap().into_iter().find(|i| i.kind == kind && i.name == name).unwrap()
}

fn fact<'a>(facts: &'a [(String, String)], key: &str) -> &'a str {
    &facts.iter().find(|(k, _)| k == key).unwrap().1
}

#[test]
fn skill_detail_shows_where_it_lands_its_files_and_its_full_skill_md() {
    let detail = load(&item(Kind::Skill, "bash-skill"), &[Target::Claude, Target::Codex], Path::new("/h")).unwrap();
    assert_eq!(fact(&detail.facts, "path"), "bash-skill");
    assert_eq!(fact(&detail.facts, "→ claude"), "~/.claude/skills/bash-skill");
    assert_eq!(fact(&detail.facts, "→ codex"), "~/.codex/skills/bash-skill");
    assert_eq!(detail.files, vec!["SKILL.md", "install.sh"]);
    assert_eq!(detail.body, vec!["---", "name: bash-skill", "description: Bash rules", "---", "body"]);
}

#[test]
fn command_detail_shows_the_prompt_users_will_run() {
    let detail = load(&item(Kind::Command, "fact-check"), &[Target::Codex], Path::new("/h")).unwrap();
    assert_eq!(fact(&detail.facts, "→ codex"), "~/.codex/prompts/fact-check.md");
    assert_eq!(detail.body[0], "# /fact-check");
    assert!(detail.files.is_empty());
}

#[test]
fn detail_warns_before_install_that_codex_skips_agents() {
    let detail = load(&item(Kind::Agent, "code-reviewer"), &[Target::Claude, Target::Codex], Path::new("/h")).unwrap();
    assert_eq!(fact(&detail.facts, "→ claude"), "~/.claude/agents/code-reviewer.md");
    assert_eq!(fact(&detail.facts, "→ codex"), "not supported by codex, skipped");
    assert_eq!(detail.body.last().map(String::as_str), Some("You review."));
}

#[test]
fn plugin_detail_shows_the_claude_install_command() {
    let detail = load(&item(Kind::Plugin, "night-shift"), &[Target::Claude], Path::new("/h")).unwrap();
    assert_eq!(fact(&detail.facts, "→ claude"), "claude plugin install night-shift@diego-market");
    assert_eq!(detail.body, vec!["Night agents plugin"]);
}
