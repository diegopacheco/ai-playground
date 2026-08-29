use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_monitor-stats");

fn run(args: &[&str]) -> (String, bool) {
    let out = Command::new(BIN).args(args).output().expect("binary runs");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.success(),
    )
}

#[test]
fn reports_the_four_requested_facts_for_every_attached_monitor() {
    let (stdout, ok) = run(&[]);
    assert!(
        ok,
        "expected at least one monitor on this machine:\n{stdout}"
    );
    let cards = stdout.matches("Brand").count();
    assert!(cards >= 1);
    assert_eq!(stdout.matches("Size").count(), cards);
    assert_eq!(stdout.matches("Max resolution").count(), cards);
    assert_eq!(stdout.matches("Max refresh rate").count(), cards);
}

#[test]
fn names_a_real_panel_rather_than_a_numeric_display_id() {
    let (stdout, _) = run(&[]);
    assert!(!stdout.contains("Unknown Display"));
    assert!(
        stdout.contains('"'),
        "a physical size in inches is missing:\n{stdout}"
    );
}

#[test]
fn json_mode_emits_one_object_per_monitor_with_the_same_count() {
    let (text, _) = run(&[]);
    let (json, ok) = run(&["--json"]);
    assert!(ok);
    assert_eq!(
        json.matches("\"brand\"").count(),
        text.matches("Brand").count()
    );
    assert!(json.starts_with('['));
    assert!(json.trim_end().ends_with(']'));
}

#[test]
fn rejects_unknown_flags_instead_of_silently_printing_a_report() {
    let (stdout, ok) = run(&["--nope"]);
    assert!(!ok);
    assert!(stdout.is_empty());
}

#[test]
fn help_and_version_succeed_without_touching_the_hardware() {
    assert!(run(&["--help"]).1);
    assert!(run(&["--version"]).0.contains(env!("CARGO_PKG_VERSION")));
}
