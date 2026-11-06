use ram::cli::{Action, parse};
use ram::fetch::DEFAULT_REPO;
use ram::kind::Target;

fn args(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

#[test]
fn no_arguments_opens_the_picker_on_the_default_repo() {
    let cli = parse(&[]).unwrap();
    assert_eq!(cli.action, Action::Interactive);
    assert_eq!(cli.repo, DEFAULT_REPO);
}

#[test]
fn install_needs_explicit_target_so_nothing_lands_in_the_wrong_home() {
    assert!(parse(&args("install bash-skill")).is_err());
    let cli = parse(&args("install bash-skill do --target both --repo /tmp/x")).unwrap();
    assert_eq!(cli.repo, "/tmp/x");
    assert_eq!(
        cli.action,
        Action::Install { selectors: vec!["bash-skill".into(), "do".into()], targets: vec![Target::Claude, Target::Codex] }
    );
}

#[test]
fn typos_are_rejected_instead_of_silently_ignored() {
    assert!(parse(&args("instal bash-skill")).is_err());
    assert!(parse(&args("install x --target vscode")).is_err());
    assert!(parse(&args("--force")).is_err());
}
