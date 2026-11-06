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
}

impl Profile {
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty() && self.agents.is_empty() && self.commands.is_empty() && self.mcps.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub source: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agent {
    pub name: String,
    pub source: String,
    pub description: Option<String>,
    pub model: Option<String>,
    pub tools: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub name: String,
    pub source: String,
    pub description: Option<String>,
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
    pub transport: Transport,
    pub description: Option<String>,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub url: Option<String>,
    pub headers: BTreeMap<String, String>,
}
