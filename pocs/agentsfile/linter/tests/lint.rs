use std::fs;
use std::path::{Path, PathBuf};

use agentsfile_lint::warnings::{is_kebab_case, looks_like_secret};
use agentsfile_lint::{Level, lint, lint_file};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spec/fixtures")
}

fn codes(src: &str) -> Vec<&'static str> {
    lint(src, Path::new(".")).into_iter().map(|d| d.code).collect()
}

#[test]
fn first_linter_error_matches_the_sdk_error_for_every_invalid_fixture() {
    for entry in fs::read_dir(fixtures().join("invalid")).unwrap() {
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
fn valid_fixtures_have_no_errors() {
    for name in ["full.agentsfile", "minimal.agentsfile"] {
        let diagnostics = lint_file(&fixtures().join("valid").join(name)).unwrap();
        assert!(diagnostics.iter().all(|d| d.level == Level::Warning), "{name}: {diagnostics:?}");
    }
}

#[test]
fn reports_every_schema_error_not_only_the_first() {
    let src = r#"version = "2"
profile "dev" {
  skill "a" {
  }
  mcp "m" {
    transport = "http"
    color = "red"
  }
}
"#;
    assert_eq!(codes(src), ["E006", "E008", "E009", "E002"]);
}

#[test]
fn syntax_error_stops_linting_with_one_e001() {
    let diagnostics = lint("version = \"1\"\nprofile \"dev\" {\n", Path::new("."));
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "E001");
}

#[test]
fn warns_on_empty_profile() {
    assert_eq!(codes("version = \"1\"\nprofile \"dev\" {\n  description = \"x\"\n}\n"), ["W001"]);
}

#[test]
fn warns_on_missing_local_source_but_not_on_urls_or_existing_paths() {
    let src = r#"version = "1"
profile "dev" {
  skill "a" { source = "./missing-skill" }
  skill "b" { source = "https://github.com/acme/skills/b" }
  skill "c" { source = "valid/full.agentsfile" }
}
"#;
    let diagnostics = lint(src, &fixtures());
    let w002: Vec<String> = diagnostics.iter().filter(|d| d.code == "W002").map(|d| d.message.clone()).collect();
    assert_eq!(w002, ["source \"./missing-skill\" does not exist"]);
}

#[test]
fn warns_on_hardcoded_secrets_and_accepts_env_references() {
    let src = r#"version = "1"
profile "dev" {
  mcp "m" {
    transport = "stdio"
    command = "x"
    env = { API_KEY = "abc123", OTHER_TOKEN = "${OTHER_TOKEN}", LOG = "debug" }
  }
}
"#;
    let diagnostics = lint(src, Path::new("."));
    let w003: Vec<&str> = diagnostics.iter().filter(|d| d.code == "W003").map(|d| d.message.as_str()).collect();
    assert_eq!(w003, ["env \"API_KEY\" looks like a hardcoded secret, use ${NAME}"]);
}

#[test]
fn secret_detection_rules() {
    assert!(looks_like_secret("GITHUB_TOKEN", "ghp_123"));
    assert!(looks_like_secret("Authorization", "Bearer abc"));
    assert!(looks_like_secret("ANYTHING", "sk-live-1"));
    assert!(!looks_like_secret("Authorization", "Bearer ${TOKEN}"));
    assert!(!looks_like_secret("LOG_LEVEL", "info"));
}

#[test]
fn warns_on_names_that_are_not_kebab_case() {
    assert_eq!(codes("version = \"1\"\nprofile \"Dev\" {\n  skill \"code_review\" { source = \"https://x\" }\n}\n"), ["W004", "W004"]);
    assert!(is_kebab_case("code-review-2"));
    assert!(!is_kebab_case("-lead"));
    assert!(!is_kebab_case("double--dash"));
}

#[test]
fn warns_on_attributes_the_transport_ignores() {
    let src = r#"version = "1"
profile "dev" {
  mcp "m" {
    transport = "http"
    url = "https://x"
    command = "npx"
  }
}
"#;
    let diagnostics = lint(src, Path::new("."));
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "W005");
    assert_eq!(diagnostics[0].message, "attribute \"command\" is ignored with http transport");
    assert_eq!((diagnostics[0].line, diagnostics[0].col), (6, 5));
}

#[test]
fn renders_path_line_col_level_and_code() {
    let diagnostics = lint("version = \"1\"\nprofile \"dev\" {\n}\n", Path::new("."));
    assert_eq!(diagnostics[0].render("Agentsfile"), "Agentsfile:2:1: warning[W001]: profile \"dev\" declares nothing");
}
