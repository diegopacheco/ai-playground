#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Factoryfile {
    pub version: String,
    pub factories: Vec<Factory>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Factory {
    pub name: String,
    pub description: Option<String>,
    pub entry: Option<String>,
    pub agents: Vec<Agent>,
    pub stages: Vec<Stage>,
    pub loops: Vec<Loop>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agent {
    pub name: String,
    pub description: Option<String>,
    pub role: Option<String>,
    pub model: Option<String>,
    pub tools: Vec<String>,
    pub source: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    pub name: String,
    pub description: Option<String>,
    pub agents: Vec<String>,
    pub next: Vec<String>,
    pub artifacts: Vec<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loop {
    pub name: String,
    pub description: Option<String>,
    pub from: String,
    pub to: String,
    pub until: Option<String>,
    pub max: Option<u32>,
}

impl Factory {
    pub fn stage(&self, name: &str) -> Option<&Stage> {
        self.stages.iter().find(|s| s.name == name)
    }

    pub fn agent(&self, name: &str) -> Option<&Agent> {
        self.agents.iter().find(|a| a.name == name)
    }
}
