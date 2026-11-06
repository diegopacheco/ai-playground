use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    Skill,
    Command,
    Agent,
    Plugin,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Skill, Kind::Command, Kind::Agent, Kind::Plugin];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Skill => "skill",
            Kind::Command => "command",
            Kind::Agent => "agent",
            Kind::Plugin => "plugin",
        }
    }

    pub fn parse(text: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|kind| kind.label() == text)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Claude,
    Codex,
}

impl Target {
    pub fn label(self) -> &'static str {
        match self {
            Target::Claude => "claude",
            Target::Codex => "codex",
        }
    }

    pub fn parse_many(text: &str) -> Option<Vec<Target>> {
        match text {
            "claude" => Some(vec![Target::Claude]),
            "codex" => Some(vec![Target::Codex]),
            "both" => Some(vec![Target::Claude, Target::Codex]),
            _ => None,
        }
    }

    pub fn supports(self, kind: Kind) -> bool {
        !(self == Target::Codex && matches!(kind, Kind::Agent | Kind::Plugin))
    }

    pub fn dir(self, home: &Path, kind: Kind) -> PathBuf {
        let base = match self {
            Target::Claude => home.join(".claude"),
            Target::Codex => home.join(".codex"),
        };
        match (self, kind) {
            (_, Kind::Skill) => base.join("skills"),
            (Target::Claude, Kind::Command) => base.join("commands"),
            (Target::Codex, Kind::Command) => base.join("prompts"),
            (_, Kind::Agent) => base.join("agents"),
            (_, Kind::Plugin) => base.join("plugins"),
        }
    }
}
