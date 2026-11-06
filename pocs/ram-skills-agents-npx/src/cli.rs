use crate::error::{Result, fail};
use crate::fetch::DEFAULT_REPO;
use crate::kind::Target;

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Interactive,
    List,
    Install { selectors: Vec<String>, targets: Vec<Target> },
    Help,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Cli {
    pub action: Action,
    pub repo: String,
}

pub const USAGE: &str = "usage:
  ram                                      interactive picker
  ram list                                 list everything in the repo
  ram install <name|kind:name>... --target claude|codex|both
options:
  --repo <url|path>                        source repo (default https://github.com/diegopacheco/agent-skills)
  -h, --help                               show this help";

pub fn parse(args: &[String]) -> Result<Cli> {
    let mut repo = DEFAULT_REPO.to_string();
    let mut targets = None;
    let mut positional = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Cli { action: Action::Help, repo }),
            "--repo" => repo = value(&mut iter, "--repo")?,
            "--target" | "-t" => {
                let text = value(&mut iter, "--target")?;
                targets = Some(Target::parse_many(&text).ok_or(format!("unknown target {text}, use claude|codex|both"))?);
            }
            flag if flag.starts_with('-') => return fail(format!("unknown option {flag}")),
            _ => positional.push(arg.clone()),
        }
    }
    let action = match positional.split_first() {
        None => Action::Interactive,
        Some((command, rest)) if command == "list" && rest.is_empty() => Action::List,
        Some((command, rest)) if command == "install" => {
            if rest.is_empty() {
                return fail("install needs at least one name");
            }
            let targets = targets.ok_or("install needs --target claude|codex|both")?;
            Action::Install { selectors: rest.to_vec(), targets }
        }
        Some((command, _)) => return fail(format!("unknown command {command}")),
    };
    Ok(Cli { action, repo })
}

fn value<'a>(iter: &mut impl Iterator<Item = &'a String>, flag: &str) -> Result<String> {
    iter.next().cloned().ok_or_else(|| format!("{flag} needs a value").into())
}
