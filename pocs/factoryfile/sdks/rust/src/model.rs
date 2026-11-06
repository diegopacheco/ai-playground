use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Factoryfile {
    pub version: String,
    pub factories: Vec<Factory>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Factory {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub entry: Option<String>,
    pub flow: Option<String>,
    pub backlog: Option<String>,
    pub agents: Vec<Agent>,
    pub stages: Vec<Stage>,
    pub loops: Vec<Loop>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agent {
    pub name: String,
    pub description: Option<String>,
    pub role: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub tools: Vec<String>,
    pub source: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Stage {
    pub name: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub agents: Vec<String>,
    pub run: Option<String>,
    pub script: Option<String>,
    pub next: Vec<String>,
    pub artifacts: Vec<String>,
    pub output: Option<String>,
    pub check: Option<String>,
    pub verdict: Option<BTreeMap<String, String>>,
    pub source: Option<String>,
    pub content: Option<String>,
    pub skills: Vec<String>,
    pub tasks: Option<String>,
    pub parallel: Option<u32>,
    pub visits: Option<u32>,
    pub end: Option<String>,
    pub at: Vec<u32>,
    pub routes: Vec<Route>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Route {
    pub outcome: String,
    pub to: String,
    pub label: Option<String>,
    pub reason: Option<String>,
    pub cycle: bool,
    pub exhausted: Option<Box<Exhausted>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Exhausted {
    pub to: String,
    pub label: Option<String>,
    pub reason: Option<String>,
    pub exhausted: Option<Box<Exhausted>>,
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

impl Route {
    pub fn fallbacks(&self) -> Vec<&Exhausted> {
        let mut out = Vec::new();
        let mut current = self.exhausted.as_deref();
        while let Some(e) = current {
            out.push(e);
            current = e.exhausted.as_deref();
        }
        out
    }
}
