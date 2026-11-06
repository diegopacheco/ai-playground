use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Agentsfile {
    pub version: String,
    pub profiles: Vec<Profile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Profile {
    pub name: String,
    pub description: Option<String>,
    pub skills: Vec<Skill>,
    pub agents: Vec<Agent>,
    pub commands: Vec<Command>,
    pub mcps: Vec<Mcp>,
    pub plugins: Vec<Plugin>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: Option<String>,
    pub source: Option<String>,
    pub content: Option<String>,
    pub files: Vec<SkillFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillFile {
    pub path: String,
    pub source: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agent {
    pub name: String,
    pub description: Option<String>,
    pub model: Option<String>,
    pub tools: Vec<String>,
    pub source: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub name: String,
    pub description: Option<String>,
    pub argument_hint: Option<String>,
    pub model: Option<String>,
    pub tools: Vec<String>,
    pub source: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    Stdio,
    Http,
    Sse,
}

impl Transport {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "stdio" => Some(Transport::Stdio),
            "http" => Some(Transport::Http),
            "sse" => Some(Transport::Sse),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Transport::Stdio => "stdio",
            Transport::Http => "http",
            Transport::Sse => "sse",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mcp {
    pub name: String,
    pub description: Option<String>,
    pub source: Option<String>,
    pub transport: Option<Transport>,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub url: Option<String>,
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plugin {
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub author: Option<String>,
    pub source: Option<String>,
    pub skills: Vec<Skill>,
    pub agents: Vec<Agent>,
    pub commands: Vec<Command>,
    pub mcps: Vec<Mcp>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Items {
    pub skills: Vec<Skill>,
    pub agents: Vec<Agent>,
    pub commands: Vec<Command>,
    pub mcps: Vec<Mcp>,
    pub plugins: Vec<Plugin>,
}

impl Items {
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
            && self.agents.is_empty()
            && self.commands.is_empty()
            && self.mcps.is_empty()
            && self.plugins.is_empty()
    }
}
