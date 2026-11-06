use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Render,
    RenderUi,
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub mode: Mode,
    pub path: PathBuf,
    pub factory: Option<String>,
    pub out: Option<PathBuf>,
    pub open: bool,
    pub color: bool,
}

pub const USAGE: &str = "usage:
  factory-cli --render    [path] [--factory NAME] [--no-color]
  factory-cli --render-ui [path] [--factory NAME] [--out FILE] [--no-open]

path defaults to ./Factoryfile";

pub fn parse(raw: &[String]) -> Result<Args, String> {
    let mut args = Args {
        mode: Mode::Help,
        path: PathBuf::from("Factoryfile"),
        factory: None,
        out: None,
        open: true,
        color: std::env::var_os("NO_COLOR").is_none(),
    };
    let mut path = None;
    let mut it = raw.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--render" => args.mode = Mode::Render,
            "--render-ui" => args.mode = Mode::RenderUi,
            "--help" | "-h" => return Ok(Args { mode: Mode::Help, ..args }),
            "--factory" => args.factory = Some(value(&mut it, arg)?),
            "--out" => args.out = Some(PathBuf::from(value(&mut it, arg)?)),
            "--no-open" => args.open = false,
            "--no-color" => args.color = false,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if let Some(p) = path {
        args.path = p;
    }
    Ok(args)
}

fn value<'a>(it: &mut impl Iterator<Item = &'a String>, flag: &str) -> Result<String, String> {
    it.next().cloned().ok_or_else(|| format!("{flag} needs a value"))
}
