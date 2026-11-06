use std::fs;
use std::path::PathBuf;

use agentsfile::{Agentsfile, Transport, parse, parse_file};

fn fixtures(kind: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/fixtures").join(kind)
}

fn full() -> Agentsfile {
    parse_file(fixtures("valid").join("full.agentsfile")).unwrap()
}

#[test]
fn every_invalid_fixture_fails_with_the_exact_spec_error() {
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for entry in fs::read_dir(fixtures("invalid")).unwrap() {
        let path = entry.unwrap().path();
        let src = fs::read_to_string(&path).unwrap();
        let expected = src.lines().next().unwrap().strip_prefix("# expect: ").unwrap();
        let actual = parse(&src).err().map(|e| e.to_string()).unwrap_or_default();
        if actual != expected {
            mismatches.push(format!("{}: expected [{expected}] got [{actual}]", path.display()));
        }
        checked += 1;
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    assert_eq!(checked, 51);
}

#[test]
fn minimal_file_has_no_profiles() {
    let file = parse_file(fixtures("valid").join("minimal.agentsfile")).unwrap();
    assert_eq!(file.version, "1");
    assert!(file.profiles.is_empty());
}

#[test]
fn full_file_keeps_profile_and_item_order() {
    let file = full();
    let names: Vec<&str> = file.profiles.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["dev", "ci"]);
    let skills: Vec<&str> = file.profiles[0].skills.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(skills, ["code-review", "release-notes"]);
    let mcps: Vec<&str> = file.profiles[1].mcps.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(mcps, ["docs", "events", "shared"]);
    let plugins: Vec<&str> = file.profiles[1].plugins.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(plugins, ["toolkit", "security"]);
}

#[test]
fn inline_skill_carries_its_whole_body_and_extra_files() {
    let file = full();
    let review = &file.profiles[0].skills[0];
    assert_eq!(review.source, None);
    assert_eq!(review.description.as_deref(), Some("Reviews the current diff"));
    assert_eq!(
        review.content.as_deref(),
        Some("# Code review\nRead the diff and list bugs first.\n\nThen list style issues.")
    );
    assert_eq!(review.files.len(), 2);
    assert_eq!(review.files[0].path, "scripts/diff.sh");
    assert_eq!(review.files[0].content.as_deref(), Some("#!/usr/bin/env bash\ngit diff --stat"));
    assert_eq!(review.files[0].source, None);
    assert_eq!(review.files[1].path, "checklist.md");
    assert_eq!(review.files[1].source.as_deref(), Some("./skills/code-review/checklist.md"));
    assert_eq!(review.files[1].content, None);

    let notes = &file.profiles[0].skills[1];
    assert_eq!(notes.source.as_deref(), Some("https://github.com/acme/skills/release-notes"));
    assert!(notes.content.is_none() && notes.files.is_empty());
}

#[test]
fn inline_agent_and_command_keep_their_prompts() {
    let file = full();
    let tester = &file.profiles[0].agents[0];
    assert_eq!(tester.model.as_deref(), Some("claude-sonnet-5-5"));
    assert_eq!(tester.tools, ["Read", "Grep", "Bash"]);
    assert_eq!(tester.content.as_deref(), Some("You write tests first.\n  Keep each test small."));
    assert_eq!(tester.source, None);

    let ship = &file.profiles[0].commands[0];
    assert_eq!(ship.argument_hint.as_deref(), Some("<branch>"));
    assert_eq!(ship.model.as_deref(), Some("claude-haiku-5-5"));
    assert_eq!(ship.tools, ["Bash"]);
    assert_eq!(ship.content.as_deref(), Some("Run the tests, then open a PR for $ARGUMENTS."));

    let fix = &file.profiles[1].commands[0];
    assert_eq!(fix.source.as_deref(), Some("./commands/fix.md"));
    assert!(fix.content.is_none() && fix.tools.is_empty() && fix.argument_hint.is_none());
}

#[test]
fn mcp_servers_are_inline_or_referenced() {
    let file = full();
    let github = &file.profiles[0].mcps[0];
    assert_eq!(github.transport, Some(Transport::Stdio));
    assert_eq!(github.command.as_deref(), Some("npx"));
    assert_eq!(github.args, ["-y", "@modelcontextprotocol/server-github"]);
    assert_eq!(github.env.get("GITHUB_TOKEN").map(String::as_str), Some("${GITHUB_TOKEN}"));
    assert_eq!(github.env.get("LOG_LEVEL").map(String::as_str), Some("info"));

    let ci = &file.profiles[1];
    assert_eq!(ci.mcps[0].transport, Some(Transport::Http));
    assert_eq!(ci.mcps[0].headers.get("Authorization").map(String::as_str), Some("Bearer ${DOCS_TOKEN}"));
    assert_eq!(ci.mcps[1].transport, Some(Transport::Sse));
    assert_eq!(ci.mcps[2].source.as_deref(), Some("./mcp/shared.json"));
    assert_eq!(ci.mcps[2].transport, None);
    assert!(ci.mcps[2].args.is_empty() && ci.mcps[2].env.is_empty() && ci.mcps[2].headers.is_empty());
}

#[test]
fn plugins_are_inline_bundles_or_referenced() {
    let file = full();
    let toolkit = &file.profiles[1].plugins[0];
    assert_eq!(toolkit.version.as_deref(), Some("1.2.0"));
    assert_eq!(toolkit.author.as_deref(), Some("Platform team"));
    assert_eq!(toolkit.description.as_deref(), Some("Shared team tooling"));
    assert_eq!(toolkit.source, None);
    assert_eq!(toolkit.skills[0].content.as_deref(), Some("Run the linter and fix every error."));
    assert_eq!(toolkit.agents[0].source.as_deref(), Some("./agents/fixer.md"));
    assert_eq!(toolkit.commands[0].name, "fix");
    assert_eq!(toolkit.mcps[0].url.as_deref(), Some("https://mcp.acme.dev/search"));

    let security = &file.profiles[1].plugins[1];
    assert_eq!(security.source.as_deref(), Some("https://github.com/acme/plugins/security"));
    assert!(security.skills.is_empty() && security.agents.is_empty() && security.commands.is_empty() && security.mcps.is_empty());
}

#[test]
fn quoted_strings_decode_escapes_and_optional_fields_default() {
    let file = full();
    let ci = &file.profiles[1];
    assert_eq!(ci.description, None);
    let reviewer = &ci.agents[0];
    assert_eq!(reviewer.description.as_deref(), Some("Says \"no\" to\tsloppy code"));
    assert!(reviewer.tools.is_empty() && reviewer.model.is_none() && reviewer.content.is_none());
}

#[test]
fn heredoc_edge_cases() {
    let file = parse_file(fixtures("valid").join("heredoc.agentsfile")).unwrap();
    assert_eq!(file.version, "1");
    let edge = &file.profiles[0];
    assert_eq!(edge.description.as_deref(), Some(""));
    assert_eq!(edge.skills[0].content.as_deref(), Some("first\n  second\n"));
    assert_eq!(edge.skills[1].content.as_deref(), Some("\"quoted\" \\n ${HOME} # not a comment"));
}

#[test]
fn columns_count_characters_not_bytes() {
    let err = parse("version = \"1\"\nprofile \"é\" { @ }").unwrap_err();
    assert_eq!(err.to_string(), "line 2, col 15: unexpected character '@'");
}

#[test]
fn positions_after_a_heredoc_follow_its_closing_line() {
    let err = parse("version = <<V\n1\nV\n  @").unwrap_err();
    assert_eq!(err.to_string(), "line 4, col 3: unexpected character '@'");
}

#[test]
fn same_name_in_different_kinds_is_allowed() {
    let src = "version = \"1\"\nprofile \"p\" {\n skill \"x\" { source = \"a\" }\n agent \"x\" { source = \"b\" }\n plugin \"x\" { source = \"c\" }\n}";
    let file = parse(src).unwrap();
    assert_eq!(file.profiles[0].skills[0].name, file.profiles[0].plugins[0].name);
}
