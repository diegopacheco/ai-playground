use std::fs;
use std::path::PathBuf;

use std::collections::BTreeMap;

use factoryfile::{Exhausted, Factory, Factoryfile, Route, parse, parse_file};

const INVALID_FIXTURES: usize = 82;

fn fixtures(kind: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/fixtures").join(kind)
}

fn full() -> Factoryfile {
    parse_file(fixtures("valid").join("full.factoryfile")).unwrap()
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
    assert_eq!(checked, INVALID_FIXTURES);
}

#[test]
fn minimal_file_has_no_factories() {
    let file = parse_file(fixtures("valid").join("minimal.factoryfile")).unwrap();
    assert_eq!(file.version, "1");
    assert!(file.factories.is_empty());
}

#[test]
fn factories_and_items_keep_document_order() {
    let file = full();
    let names: Vec<&str> = file.factories.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["scrum", "solo", "empty"]);
    let stages: Vec<&str> = file.factories[0].stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(stages, ["refinement", "planning", "build", "review", "retro"]);
    let loops: Vec<&str> = file.factories[0].loops.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(loops, ["sprint", "daily"]);
}

#[test]
fn explicit_entry_wins_and_missing_entry_falls_back_to_the_first_stage() {
    let file = full();
    assert_eq!(file.factories[0].entry.as_deref(), Some("refinement"));
    assert_eq!(file.factories[1].entry.as_deref(), Some("code"));
    assert_eq!(file.factories[2].entry, None);
}

#[test]
fn inline_agent_carries_its_whole_prompt_and_source_agent_points_away() {
    let file = full();
    let po = file.factories[0].agent("product-owner").unwrap();
    assert_eq!(po.role.as_deref(), Some("product owner"));
    assert_eq!(po.model.as_deref(), Some("claude-opus-5-5"));
    assert_eq!(po.tools, ["Read", "Write"]);
    assert_eq!(po.source, None);
    assert_eq!(
        po.content.as_deref(),
        Some("You keep the backlog ordered by value.\n\nEvery story has acceptance criteria.")
    );
    let dev = file.factories[0].agent("developer").unwrap();
    assert_eq!(dev.source.as_deref(), Some("./agents/developer.md"));
    assert_eq!(dev.content, None);
    assert_eq!(dev.model, None);
}

#[test]
fn stages_hold_the_graph_edges_and_their_artifacts() {
    let file = full();
    let scrum = &file.factories[0];
    let build = scrum.stage("build").unwrap();
    assert_eq!(build.agents, ["developer"]);
    assert_eq!(build.next, ["review", "retro"]);
    assert_eq!(build.artifacts, ["src/", "tests/"]);
    let review = scrum.stage("review").unwrap();
    assert!(review.next.is_empty(), "a stage without next ends the factory");
    assert!(scrum.stage("retro").unwrap().agents.is_empty());
    assert_eq!(scrum.stage("planning").unwrap().content.as_deref(), Some("Pick the stories for the sprint goal."));
}

#[test]
fn loops_are_back_edges_with_an_exit_condition_and_a_bound() {
    let file = full();
    let sprint = &file.factories[0].loops[0];
    assert_eq!((sprint.from.as_str(), sprint.to.as_str()), ("review", "planning"));
    assert_eq!(sprint.until.as_deref(), Some("the backlog is empty"));
    assert_eq!(sprint.max, Some(26));
    let daily = &file.factories[0].loops[1];
    assert_eq!((daily.from.as_str(), daily.to.as_str()), ("build", "build"));
    assert_eq!((daily.until.clone(), daily.max), (None, None));
}

#[test]
fn heredocs_strip_shared_indent_and_keep_text_raw() {
    let file = parse_file(fixtures("valid").join("heredoc.factoryfile")).unwrap();
    assert_eq!(file.version, "1");
    let edge = &file.factories[0];
    assert_eq!(edge.description.as_deref(), Some(""));
    assert_eq!(edge.agents[0].content.as_deref(), Some("first\n  second\n"));
    assert_eq!(edge.agents[1].content.as_deref(), Some("\"quoted\" \\n ${HOME} # not a comment"));
    assert_eq!(edge.stages[0].agents, ["tabs", "raw"]);
}

#[test]
fn columns_count_characters_not_bytes() {
    let err = parse("version = \"1\"\nfactory \"é\" { owner = \"x\" }\n").unwrap_err();
    assert_eq!(err.to_string(), "line 2, col 15: unknown attribute \"owner\" in factory");
}

#[test]
fn a_cycle_through_next_is_legal_for_a_parser() {
    let src = "version = \"1\"\nfactory \"x\" {\n  stage \"a\" { agents = [] next = [\"b\"] }\n  stage \"b\" { agents = [] next = [\"a\"] }\n}\n";
    let file = parse(src).unwrap();
    assert_eq!(file.factories[0].stages[1].next, ["a"]);
}

fn gate() -> Factory {
    parse_file(fixtures("valid").join("routes.factoryfile")).unwrap().factories.remove(0)
}

fn text(s: &str) -> Option<String> {
    Some(s.to_string())
}

#[test]
fn v11_factory_carries_title_flow_backlog_and_agent_providers() {
    let gate = gate();
    assert_eq!(gate.title, text("Approval Gate"));
    assert_eq!(gate.flow, text("pull"));
    assert_eq!(gate.backlog, text("optional"));
    assert_eq!(gate.agent("maker").unwrap().provider, text("codex"));
    assert_eq!(gate.agent("approver").unwrap().provider, text("claude"));
    assert_eq!(full().factories[0].title, None, "a v1 factory has no title");
}

#[test]
fn v11_command_stages_need_no_agents_and_keep_their_run_and_end() {
    let gate = gate();
    let test = gate.stage("test").unwrap();
    assert_eq!(test.run, text("test"));
    assert!(test.agents.is_empty());
    assert_eq!(test.label, text("Test"));
    assert_eq!(gate.stage("ship").unwrap().script, text("scripts/ship.sh"));
    let done = gate.stage("done").unwrap();
    assert_eq!(done.end, text("succeeded"));
    assert_eq!(done.at, [1, 1]);
    assert!(full().factories[0].stage("build").unwrap().at.is_empty(), "a stage without at");
}

#[test]
fn v11_agent_stages_carry_their_prompt_file_skills_output_check_and_verdict() {
    let gate = gate();
    let implement = gate.stage("implement").unwrap();
    assert_eq!(implement.source, text("prompts/implement.md"));
    assert_eq!(implement.skills, ["skills/tdd.md", "skills/git.md"]);
    assert_eq!(implement.visits, Some(4));
    let approve = gate.stage("approve").unwrap();
    assert_eq!(approve.output, text("approval.json"));
    assert_eq!(approve.check, text("verdict"));
    let verdict: BTreeMap<String, String> = [("field", "approved"), ("yes", "approved"), ("no", "rejected"), ("list", "reasons"), ("notes", "notes")]
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .into();
    assert_eq!(approve.verdict, Some(verdict));
    let build = gate.stage("build").unwrap();
    assert_eq!(build.tasks, text("split"));
    assert_eq!(build.parallel, Some(10));
    assert_eq!(build.verdict, None);
}

#[test]
fn v11_routes_branch_by_outcome_and_fall_back_when_their_target_is_exhausted() {
    let gate = gate();
    let route = |outcome: &str, to: &str| Route { outcome: outcome.to_string(), to: to.to_string(), ..Route::default() };
    assert_eq!(
        gate.stage("test").unwrap().routes,
        [
            Route { label: text("pass"), ..route("passed", "approve") },
            Route { label: text("fail"), reason: text("the tests failed"), ..route("failed", "implement") },
        ]
    );
    assert_eq!(gate.stage("approve").unwrap().routes[1], Route { cycle: true, ..route("rejected", "implement") });
    let last = Exhausted { to: "done".to_string(), label: text("last fallback"), ..Exhausted::default() };
    let first = Exhausted {
        to: "approve".to_string(),
        label: text("first fallback"),
        reason: text("test already ran as often as allowed"),
        exhausted: Some(Box::new(last)),
    };
    assert_eq!(
        gate.stage("implement").unwrap().routes,
        [Route { exhausted: Some(Box::new(first)), ..route("next", "test") }]
    );
    assert!(gate.stage("split").unwrap().routes.is_empty(), "a stage with only next has no routes");
}
