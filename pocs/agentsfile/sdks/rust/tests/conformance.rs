use std::fs;
use std::path::PathBuf;

use agentsfile::{Transport, parse, parse_file};

fn fixtures(kind: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/fixtures").join(kind)
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
    assert_eq!(checked, 30);
}

#[test]
fn minimal_file_has_no_profiles() {
    let file = parse_file(fixtures("valid").join("minimal.agentsfile")).unwrap();
    assert_eq!(file.version, "1");
    assert!(file.profiles.is_empty());
}

#[test]
fn full_file_keeps_profile_and_item_order() {
    let file = parse_file(fixtures("valid").join("full.agentsfile")).unwrap();
    let names: Vec<&str> = file.profiles.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["dev", "ci"]);
    let dev = &file.profiles[0];
    let skills: Vec<&str> = dev.skills.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(skills, ["code-review", "release-notes"]);
    let ci = &file.profiles[1];
    let mcps: Vec<&str> = ci.mcps.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(mcps, ["docs", "events"]);
}

#[test]
fn full_file_maps_every_block_kind_to_typed_fields() {
    let file = parse_file(fixtures("valid").join("full.agentsfile")).unwrap();
    let dev = &file.profiles[0];
    assert_eq!(dev.description.as_deref(), Some("Everyday coding setup"));
    assert_eq!(dev.skills[0].source, "./skills/code-review");
    assert_eq!(dev.skills[0].description.as_deref(), Some("Reviews the current diff"));
    assert_eq!(dev.skills[1].description, None);

    let tester = &dev.agents[0];
    assert_eq!(tester.source, "./agents/tester.md");
    assert_eq!(tester.model.as_deref(), Some("claude-sonnet-5-5"));
    assert_eq!(tester.tools, ["Read", "Grep", "Bash"]);

    assert_eq!(dev.commands[0].name, "ship");
    assert_eq!(dev.commands[0].source, "./commands/ship.md");

    let github = &dev.mcps[0];
    assert_eq!(github.transport, Transport::Stdio);
    assert_eq!(github.command.as_deref(), Some("npx"));
    assert_eq!(github.args, ["-y", "@modelcontextprotocol/server-github"]);
    assert_eq!(github.env.get("GITHUB_TOKEN").map(String::as_str), Some("${GITHUB_TOKEN}"));
    assert_eq!(github.env.get("LOG_LEVEL").map(String::as_str), Some("info"));
    assert_eq!(github.url, None);
}

#[test]
fn full_file_decodes_escapes_and_applies_defaults() {
    let file = parse_file(fixtures("valid").join("full.agentsfile")).unwrap();
    let ci = &file.profiles[1];
    assert_eq!(ci.description, None);
    let reviewer = &ci.agents[0];
    assert_eq!(reviewer.description.as_deref(), Some("Says \"no\" to\tsloppy code"));
    assert!(reviewer.tools.is_empty());
    assert_eq!(reviewer.model, None);

    let docs = &ci.mcps[0];
    assert_eq!(docs.transport, Transport::Http);
    assert_eq!(docs.url.as_deref(), Some("https://mcp.acme.dev/docs"));
    assert_eq!(docs.headers.get("Authorization").map(String::as_str), Some("Bearer ${DOCS_TOKEN}"));
    assert!(docs.args.is_empty() && docs.env.is_empty() && docs.command.is_none());

    let events = &ci.mcps[1];
    assert_eq!(events.transport, Transport::Sse);
    assert!(events.headers.is_empty());
}

#[test]
fn columns_count_characters_not_bytes() {
    let err = parse("version = \"1\"\nprofile \"é\" { @ }").unwrap_err();
    assert_eq!(err.to_string(), "line 2, col 15: unexpected character '@'");
}

#[test]
fn same_name_in_different_kinds_is_allowed() {
    let src = "version = \"1\"\nprofile \"p\" {\n skill \"x\" { source = \"a\" }\n agent \"x\" { source = \"b\" }\n}";
    let file = parse(src).unwrap();
    assert_eq!(file.profiles[0].skills[0].name, file.profiles[0].agents[0].name);
}
