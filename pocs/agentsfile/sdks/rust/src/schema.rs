use std::collections::{BTreeMap, HashSet};

use crate::error::Error;
use crate::model::{Agent, Agentsfile, Command, Items, Mcp, Plugin, Profile, Skill, SkillFile, Transport};
use crate::parser::{Attribute, Block, Statement, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueKind {
    UnknownAttribute,
    UnknownBlock,
    DuplicateAttribute,
    WrongKind,
    InvalidValue,
    DuplicateName,
    MissingRequired,
    TransportRequirement,
    Conflict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    pub kind: IssueKind,
    pub error: Error,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Str,
    List,
    Map,
}

const FILE_ATTRS: &[(&str, Kind)] = &[("version", Kind::Str)];
const PROFILE_ATTRS: &[(&str, Kind)] = &[("description", Kind::Str)];
const SKILL_ATTRS: &[(&str, Kind)] = &[("description", Kind::Str), ("source", Kind::Str), ("content", Kind::Str)];
const SKILL_FILE_ATTRS: &[(&str, Kind)] = &[("source", Kind::Str), ("content", Kind::Str)];
const AGENT_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("model", Kind::Str),
    ("tools", Kind::List),
    ("source", Kind::Str),
    ("content", Kind::Str),
];
const COMMAND_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("argument-hint", Kind::Str),
    ("model", Kind::Str),
    ("tools", Kind::List),
    ("source", Kind::Str),
    ("content", Kind::Str),
];
const MCP_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("source", Kind::Str),
    ("transport", Kind::Str),
    ("command", Kind::Str),
    ("args", Kind::List),
    ("env", Kind::Map),
    ("url", Kind::Str),
    ("headers", Kind::Map),
];
const PLUGIN_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("version", Kind::Str),
    ("author", Kind::Str),
    ("source", Kind::Str),
];
const PROFILE_KINDS: &[&str] = &["skill", "agent", "command", "mcp", "plugin"];
const PLUGIN_KINDS: &[&str] = &["skill", "agent", "command", "mcp"];

struct Attrs<'a>(Vec<&'a Attribute>);

impl Attrs<'_> {
    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|a| a.name == name)
    }

    fn value(&self, name: &str) -> Option<&Value> {
        self.0.iter().find(|a| a.name == name).map(|a| &a.value)
    }

    fn str(&self, name: &str) -> Option<String> {
        match self.value(name) {
            Some(Value::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }

    fn list(&self, name: &str) -> Vec<String> {
        match self.value(name) {
            Some(Value::List(items)) => items.clone(),
            _ => Vec::new(),
        }
    }

    fn map(&self, name: &str) -> BTreeMap<String, String> {
        match self.value(name) {
            Some(Value::Map(m)) => m.clone(),
            _ => BTreeMap::new(),
        }
    }
}

pub fn validate(statements: &[Statement]) -> (Agentsfile, Vec<Issue>) {
    let mut v = Validator { issues: Vec::new() };
    let file = v.file(statements);
    (file, v.issues)
}

struct Validator {
    issues: Vec<Issue>,
}

impl Validator {
    fn report(&mut self, kind: IssueKind, line: usize, col: usize, message: String) {
        self.issues.push(Issue {
            kind,
            error: Error::new(line, col, message),
        });
    }

    fn file(&mut self, statements: &[Statement]) -> Agentsfile {
        let mut profiles: Vec<Profile> = Vec::new();
        let mut seen = HashSet::new();
        let attrs = self.walk("file", statements, FILE_ATTRS, |v, b| {
            if b.kind != "profile" {
                v.unknown_block(b, "file");
                return;
            }
            if !seen.insert(b.label.clone()) {
                v.report(IssueKind::DuplicateName, b.line, b.col, format!("duplicate profile \"{}\"", b.label));
            }
            profiles.push(v.profile(b));
        });
        if !attrs.has("version") {
            self.report(IssueKind::MissingRequired, 1, 1, "missing required attribute \"version\"".to_string());
        }
        Agentsfile {
            version: attrs.str("version").unwrap_or_default(),
            profiles,
        }
    }

    fn profile(&mut self, block: &Block) -> Profile {
        let (attrs, items) = self.container(block, PROFILE_ATTRS, PROFILE_KINDS);
        Profile {
            name: block.label.clone(),
            description: attrs.str("description"),
            skills: items.skills,
            agents: items.agents,
            commands: items.commands,
            mcps: items.mcps,
            plugins: items.plugins,
        }
    }

    fn plugin(&mut self, b: &Block) -> Plugin {
        let (attrs, items) = self.container(b, PLUGIN_ATTRS, PLUGIN_KINDS);
        self.exactly_one(b, attrs.has("source"), !items.is_empty(), "inline items");
        Plugin {
            name: b.label.clone(),
            description: attrs.str("description"),
            version: attrs.str("version"),
            author: attrs.str("author"),
            source: attrs.str("source"),
            skills: items.skills,
            agents: items.agents,
            commands: items.commands,
            mcps: items.mcps,
        }
    }

    fn container<'a>(&mut self, owner: &'a Block, allowed: &[(&str, Kind)], kinds: &[&str]) -> (Attrs<'a>, Items) {
        let mut items = Items::default();
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let attrs = self.walk(&owner.kind, &owner.body, allowed, |v, b| {
            if !kinds.contains(&b.kind.as_str()) {
                v.unknown_block(b, &owner.kind);
                return;
            }
            if !seen.insert((b.kind.clone(), b.label.clone())) {
                v.report(
                    IssueKind::DuplicateName,
                    b.line,
                    b.col,
                    format!("duplicate {} \"{}\" in {} \"{}\"", b.kind, b.label, owner.kind, owner.label),
                );
            }
            match b.kind.as_str() {
                "skill" => items.skills.push(v.skill(b)),
                "agent" => items.agents.push(v.agent(b)),
                "command" => items.commands.push(v.command(b)),
                "mcp" => items.mcps.push(v.mcp(b)),
                _ => items.plugins.push(v.plugin(b)),
            }
        });
        (attrs, items)
    }

    fn skill(&mut self, b: &Block) -> Skill {
        let mut files = Vec::new();
        let mut seen = HashSet::new();
        let attrs = self.walk("skill", &b.body, SKILL_ATTRS, |v, child| {
            if child.kind != "file" {
                v.unknown_block(child, "skill");
                return;
            }
            if !seen.insert(child.label.clone()) {
                v.report(
                    IssueKind::DuplicateName,
                    child.line,
                    child.col,
                    format!("duplicate file \"{}\" in skill \"{}\"", child.label, b.label),
                );
            }
            files.push(v.skill_file(child));
        });
        self.source_or_content(b, &attrs);
        Skill {
            name: b.label.clone(),
            description: attrs.str("description"),
            source: attrs.str("source"),
            content: attrs.str("content"),
            files,
        }
    }

    fn skill_file(&mut self, b: &Block) -> SkillFile {
        let attrs = self.leaf(b, SKILL_FILE_ATTRS);
        self.source_or_content(b, &attrs);
        SkillFile {
            path: b.label.clone(),
            source: attrs.str("source"),
            content: attrs.str("content"),
        }
    }

    fn agent(&mut self, b: &Block) -> Agent {
        let attrs = self.leaf(b, AGENT_ATTRS);
        self.source_or_content(b, &attrs);
        Agent {
            name: b.label.clone(),
            description: attrs.str("description"),
            model: attrs.str("model"),
            tools: attrs.list("tools"),
            source: attrs.str("source"),
            content: attrs.str("content"),
        }
    }

    fn command(&mut self, b: &Block) -> Command {
        let attrs = self.leaf(b, COMMAND_ATTRS);
        self.source_or_content(b, &attrs);
        Command {
            name: b.label.clone(),
            description: attrs.str("description"),
            argument_hint: attrs.str("argument-hint"),
            model: attrs.str("model"),
            tools: attrs.list("tools"),
            source: attrs.str("source"),
            content: attrs.str("content"),
        }
    }

    fn mcp(&mut self, b: &Block) -> Mcp {
        let attrs = self.leaf(b, MCP_ATTRS);
        self.exactly_one(b, attrs.has("source"), attrs.has("transport"), "\"transport\"");
        let transport = attrs.str("transport").and_then(|t| Transport::from_name(&t));
        if let Some(t) = transport {
            let needed = if t == Transport::Stdio { "command" } else { "url" };
            if !attrs.has(needed) {
                self.report(
                    IssueKind::TransportRequirement,
                    b.line,
                    b.col,
                    format!("mcp \"{}\" with {} transport requires \"{needed}\"", b.label, t.name()),
                );
            }
        }
        Mcp {
            name: b.label.clone(),
            description: attrs.str("description"),
            source: attrs.str("source"),
            transport,
            command: attrs.str("command"),
            args: attrs.list("args"),
            env: attrs.map("env"),
            url: attrs.str("url"),
            headers: attrs.map("headers"),
        }
    }

    fn source_or_content(&mut self, b: &Block, attrs: &Attrs<'_>) {
        self.exactly_one(b, attrs.has("source"), attrs.has("content"), "\"content\"");
    }

    fn exactly_one(&mut self, b: &Block, has_source: bool, has_inline: bool, inline: &str) {
        if !has_source && !has_inline {
            self.report(
                IssueKind::MissingRequired,
                b.line,
                b.col,
                format!("{} \"{}\" requires \"source\" or {inline}", b.kind, b.label),
            );
        } else if has_source && has_inline {
            self.report(
                IssueKind::Conflict,
                b.line,
                b.col,
                format!("{} \"{}\" cannot set both \"source\" and {inline}", b.kind, b.label),
            );
        }
    }

    fn leaf<'a>(&mut self, b: &'a Block, allowed: &[(&str, Kind)]) -> Attrs<'a> {
        self.walk(&b.kind, &b.body, allowed, |v, child| v.unknown_block(child, &b.kind))
    }

    fn unknown_block(&mut self, b: &Block, ctx: &str) {
        self.report(IssueKind::UnknownBlock, b.line, b.col, format!("unknown block \"{}\" in {ctx}", b.kind));
    }

    fn walk<'a>(
        &mut self,
        ctx: &str,
        body: &'a [Statement],
        allowed: &[(&str, Kind)],
        mut on_block: impl FnMut(&mut Self, &'a Block),
    ) -> Attrs<'a> {
        let mut accepted: Vec<&'a Attribute> = Vec::new();
        let mut seen = HashSet::new();
        for statement in body {
            let a = match statement {
                Statement::Block(b) => {
                    on_block(self, b);
                    continue;
                }
                Statement::Attribute(a) => a,
            };
            let Some((_, kind)) = allowed.iter().find(|(n, _)| *n == a.name) else {
                self.report(IssueKind::UnknownAttribute, a.line, a.col, format!("unknown attribute \"{}\" in {ctx}", a.name));
                continue;
            };
            if !seen.insert(a.name.clone()) {
                self.report(IssueKind::DuplicateAttribute, a.line, a.col, format!("duplicate attribute \"{}\" in {ctx}", a.name));
                continue;
            }
            if !matches_kind(&a.value, *kind) {
                self.report(IssueKind::WrongKind, a.line, a.col, format!("attribute \"{}\" must be {}", a.name, kind_name(*kind)));
                continue;
            }
            self.check_value(ctx, a);
            accepted.push(a);
        }
        Attrs(accepted)
    }

    fn check_value(&mut self, ctx: &str, a: &Attribute) {
        let Value::Str(s) = &a.value else { return };
        let message = match (ctx, a.name.as_str()) {
            ("file", "version") if s != "1" => format!("unsupported version \"{s}\""),
            ("mcp", "transport") if Transport::from_name(s).is_none() => {
                format!("invalid transport \"{s}\", expected stdio, http or sse")
            }
            _ => return,
        };
        self.report(IssueKind::InvalidValue, a.value_line, a.value_col, message);
    }
}

fn matches_kind(value: &Value, kind: Kind) -> bool {
    matches!(
        (value, kind),
        (Value::Str(_), Kind::Str) | (Value::List(_), Kind::List) | (Value::Map(_), Kind::Map)
    )
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Str => "a string",
        Kind::List => "a list",
        Kind::Map => "a map",
    }
}
