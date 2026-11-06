mod support;

use ram::catalog::{Item, scan};
use ram::kind::Kind;

fn find<'a>(items: &'a [Item], kind: Kind, name: &str) -> &'a Item {
    items.iter().find(|i| i.kind == kind && i.name == name).unwrap_or_else(|| panic!("missing {name}"))
}

#[test]
fn every_installable_kind_is_discovered_from_repo_conventions() {
    let items = scan(&support::fixture_repo()).unwrap();
    let names: Vec<(Kind, &str)> = items.iter().map(|i| (i.kind, i.name.as_str())).collect();
    assert_eq!(
        names,
        vec![
            (Kind::Skill, "bash-skill"),
            (Kind::Skill, "fact-checker"),
            (Kind::Skill, "sql-migration"),
            (Kind::Command, "do"),
            (Kind::Command, "fact-check"),
            (Kind::Agent, "code-reviewer"),
            (Kind::Plugin, "night-shift"),
        ]
    );
}

#[test]
fn skill_name_comes_from_frontmatter_so_agents_can_find_it_not_from_generic_dir() {
    let items = scan(&support::fixture_repo()).unwrap();
    let skill = find(&items, Kind::Skill, "sql-migration");
    assert_eq!(skill.path, "patterns/01-x/skill");
    assert_eq!(skill.description, "Plan migrations");
}

#[test]
fn dependencies_and_readmes_are_never_offered_as_installables() {
    let items = scan(&support::fixture_repo()).unwrap();
    assert!(items.iter().all(|i| i.name != "evil" && i.name != "README"));
}

#[test]
fn plugins_carry_their_marketplace_for_claude_plugin_install() {
    let items = scan(&support::fixture_repo()).unwrap();
    let plugin = find(&items, Kind::Plugin, "night-shift");
    assert_eq!(plugin.marketplace.as_deref(), Some("diego-market"));
    assert_eq!(plugin.path, "plugins/night-shift");
}

#[test]
fn selectors_match_by_name_or_narrow_by_kind() {
    let items = scan(&support::fixture_repo()).unwrap();
    let skill = find(&items, Kind::Skill, "fact-checker");
    assert!(skill.matches("fact-checker"));
    assert!(skill.matches("skill:fact-checker"));
    assert!(!skill.matches("command:fact-checker"));
}
