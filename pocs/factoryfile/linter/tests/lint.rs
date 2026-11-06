use std::fs;
use std::path::{Path, PathBuf};

use factoryfile_lint::warnings::{is_kebab_case, is_local_path};
use factoryfile_lint::{Level, lint, lint_file};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn codes(src: &str) -> Vec<&'static str> {
    lint(src, Path::new(".")).into_iter().map(|d| d.code).collect()
}

fn factory(body: &str) -> String {
    format!("version = \"1\"\nfactory \"f\" {{\n{body}\n}}\n")
}

#[test]
fn first_linter_error_matches_the_sdk_error_for_every_invalid_fixture() {
    for entry in fs::read_dir(root().join("spec/fixtures/invalid")).unwrap() {
        let path = entry.unwrap().path();
        let src = fs::read_to_string(&path).unwrap();
        let expected = src.lines().next().unwrap().strip_prefix("# expect: ").unwrap();
        let errors: Vec<String> = lint(&src, Path::new("."))
            .into_iter()
            .filter(|d| d.level == Level::Error)
            .map(|d| format!("line {}, col {}: {}", d.line, d.col, d.message))
            .collect();
        assert!(errors.iter().any(|e| e == expected), "{}: linter missed [{expected}] in {errors:?}", path.display());
    }
}

#[test]
fn valid_fixtures_and_the_root_factoryfile_have_no_errors() {
    let mut files = vec![root().join("Factoryfile")];
    files.extend(["full", "minimal", "heredoc"].map(|n| root().join(format!("spec/fixtures/valid/{n}.factoryfile"))));
    for path in files {
        let diagnostics = lint_file(&path).unwrap();
        assert!(diagnostics.iter().all(|d| d.level == Level::Warning), "{}: {diagnostics:?}", path.display());
    }
}

#[test]
fn the_root_factoryfile_lints_clean_so_it_can_dogfood_the_format() {
    assert_eq!(lint_file(&root().join("Factoryfile")).unwrap(), vec![]);
}

#[test]
fn every_sample_has_no_errors_and_only_xgh_breaks_the_good_practices() {
    for entry in fs::read_dir(root().join("samples")).unwrap() {
        let path = entry.unwrap().path();
        let diagnostics = lint_file(&path).unwrap();
        assert!(diagnostics.iter().all(|d| d.level == Level::Warning), "{}: {diagnostics:?}", path.display());
        let is_xgh = path.file_name().unwrap() == "xgh.factoryfile";
        assert_eq!(!diagnostics.is_empty(), is_xgh, "{}: {diagnostics:?}", path.display());
    }
}

#[test]
fn xgh_is_flagged_for_unbounded_rework_dead_stages_and_idle_testers() {
    let diagnostics = lint_file(&root().join("samples/xgh.factoryfile")).unwrap();
    let mut found: Vec<&str> = diagnostics.iter().map(|d| d.code).collect();
    found.sort();
    assert_eq!(found, ["W003", "W005", "W007", "W008"]);
}

#[test]
fn reports_every_schema_error_not_only_the_first() {
    let src = "version = \"2\"\nfactory \"f\" {\n  stage \"a\" { next = [\"ghost\"] }\n  loop \"l\" { max = \"0\" }\n}\n";
    let found = codes(src);
    for code in ["E006", "E008", "E009"] {
        assert!(found.contains(&code), "missing {code} in {found:?}");
    }
    assert_eq!(found.iter().filter(|c| **c == "E008").count(), 3, "agents, from and to are all missing: {found:?}");
}

#[test]
fn a_syntax_error_stops_the_lint_with_a_single_e001() {
    assert_eq!(codes("version = \"1\"\nfactory \"f\" {\n"), ["E001"]);
}

#[test]
fn diagnostics_are_sorted_by_position() {
    let src = factory("  stage \"B\" { agents = [\"ghost\"] }\n  agent \"Idle\" { content = \"\" }");
    let positions: Vec<(usize, usize)> = lint(&src, Path::new(".")).iter().map(|d| (d.line, d.col)).collect();
    let mut sorted = positions.clone();
    sorted.sort();
    assert_eq!(positions, sorted);
}

#[test]
fn w001_factory_without_stages() {
    assert_eq!(codes(&factory("")), ["W001"]);
}

#[test]
fn w002_local_source_must_exist_but_urls_are_not_checked() {
    let missing = factory("  agent \"a\" { source = \"./nope.md\" }\n  stage \"s\" { agents = [\"a\"] }");
    assert_eq!(codes(&missing), ["W002"]);
    let url = factory("  agent \"a\" { source = \"https://x/a.md\" }\n  stage \"s\" { agents = [\"a\"] }");
    assert!(codes(&url).is_empty());
    let existing = factory("  agent \"a\" { source = \"./agents/sdk-porter.md\" }\n  stage \"s\" { agents = [\"a\"] }");
    assert!(lint(&existing, &root()).is_empty());
}

#[test]
fn w003_stage_reachable_only_through_a_loop_is_not_flagged() {
    let src = factory(
        "  stage \"a\" { agents = [] next = [\"b\"] }\n  stage \"b\" { agents = [] }\n  stage \"c\" { agents = [] }\n  stage \"island\" { agents = [] }\n  loop \"l\" { from = \"b\" to = \"c\" max = \"2\" }",
    );
    let diagnostics = lint(&src, Path::new("."));
    let w003: Vec<&str> = diagnostics.iter().filter(|d| d.code == "W003").map(|d| d.message.as_str()).collect();
    assert_eq!(w003, ["stage \"island\" cannot be reached from the entry \"a\""]);
}

#[test]
fn w003_reachability_starts_at_the_explicit_entry() {
    let src = factory("  entry = \"b\"\n  stage \"a\" { agents = [] next = [\"b\"] }\n  stage \"b\" { agents = [] }");
    assert_eq!(codes(&src), ["W003"]);
}

#[test]
fn w004_names_must_be_kebab_case() {
    assert!(is_kebab_case("sprint-review-2"));
    for bad in ["Sprint", "sprint_review", "-a", "a-", "a--b", ""] {
        assert!(!is_kebab_case(bad), "{bad}");
    }
    assert_eq!(codes(&factory("  stage \"Build\" { agents = [] }")), ["W004"]);
}

#[test]
fn w005_agent_never_used_by_a_stage() {
    let src = factory("  agent \"idle\" { content = \"x\" }\n  stage \"s\" { agents = [] }");
    assert_eq!(codes(&src), ["W005"]);
}

#[test]
fn w006_blank_inline_content() {
    let src = factory("  agent \"a\" { content = \"  \" }\n  stage \"s\" { agents = [\"a\"] content = \"\" }");
    assert_eq!(codes(&src), ["W006", "W006"]);
}

#[test]
fn w007_next_going_back_is_an_unbounded_cycle_but_a_loop_is_not() {
    let cycle = factory("  stage \"a\" { agents = [] next = [\"b\"] }\n  stage \"b\" { agents = [] next = [\"a\"] }");
    let diagnostics = lint(&cycle, Path::new("."));
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "W007");
    assert_eq!((diagnostics[0].line, diagnostics[0].col), (4, 34));
    let looped = factory("  stage \"a\" { agents = [] next = [\"b\"] }\n  stage \"b\" { agents = [] }\n  loop \"l\" { from = \"b\" to = \"a\" max = \"3\" }");
    assert!(codes(&looped).is_empty());
}

#[test]
fn w007_a_diamond_is_not_a_cycle() {
    let src = factory(
        "  stage \"a\" { agents = [] next = [\"b\", \"c\"] }\n  stage \"b\" { agents = [] next = [\"d\"] }\n  stage \"c\" { agents = [] next = [\"d\"] }\n  stage \"d\" { agents = [] }",
    );
    assert!(codes(&src).is_empty());
}

#[test]
fn w008_loop_without_max() {
    let src = factory("  stage \"a\" { agents = [] }\n  loop \"l\" { from = \"a\" to = \"a\" }");
    assert_eq!(codes(&src), ["W008"]);
}

#[test]
fn local_paths_start_with_dot_or_slash() {
    assert!(is_local_path("./a.md") && is_local_path("/tmp/a.md"));
    assert!(!is_local_path("https://x/a.md") && !is_local_path("agents/a.md"));
}
