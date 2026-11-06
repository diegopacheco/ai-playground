use std::collections::{BTreeMap, HashSet};

use crate::error::Error;
use crate::model::{Agent, Agentsfile, Command, Mcp, Profile, Skill, Transport};
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
const SKILL_ATTRS: &[(&str, Kind)] = &[("source", Kind::Str), ("description", Kind::Str)];
const COMMAND_ATTRS: &[(&str, Kind)] = SKILL_ATTRS;
const AGENT_ATTRS: &[(&str, Kind)] = &[
    ("source", Kind::Str),
    ("description", Kind::Str),
    ("model", Kind::Str),
    ("tools", Kind::List),
];
const MCP_ATTRS: &[(&str, Kind)] = &[
    ("transport", Kind::Str),
    ("description", Kind::Str),
    ("command", Kind::Str),
    ("args", Kind::List),
    ("env", Kind::Map),
    ("url", Kind::Str),
    ("headers", Kind::Map),
];

struct Attrs<'a>(Vec<&'a Attribute>);

impl Attrs<'_> {
    fn get(&self, name: &str) -> Option<&Attribute> {
        self.0.iter().copied().find(|a| a.name == name)
    }

    fn str(&self, name: &str) -> Option<String> {
        match self.get(name).map(|a| &a.value) {
            Some(Value::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }

    fn list(&self, name: &str) -> Vec<String> {
        match self.get(name).map(|a| &a.value) {
            Some(Value::List(items)) => items.clone(),
            _ => Vec::new(),
        }
    }

    fn map(&self, name: &str) -> BTreeMap<String, String> {
        match self.get(name).map(|a| &a.value) {
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
        let version = attrs.str("version");
        if attrs.get("version").is_none() {
            self.report(IssueKind::MissingRequired, 1, 1, "missing required attribute \"version\"".to_string());
        }
        Agentsfile {
            version: version.unwrap_or_default(),
            profiles,
        }
    }

    fn profile(&mut self, block: &Block) -> Profile {
        let mut p = Profile {
            name: block.label.clone(),
            ..Profile::default()
        };
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let attrs = self.walk("profile", &block.body, PROFILE_ATTRS, |v, b| {
            let known = ["skill", "agent", "command", "mcp"].contains(&b.kind.as_str());
            if !known {
                v.unknown_block(b, "profile");
                return;
            }
            if !seen.insert((b.kind.clone(), b.label.clone())) {
                v.report(
                    IssueKind::DuplicateName,
                    b.line,
                    b.col,
                    format!("duplicate {} \"{}\" in profile \"{}\"", b.kind, b.label, block.label),
                );
            }
            match b.kind.as_str() {
                "skill" => p.skills.push(v.skill(b)),
                "agent" => p.agents.push(v.agent(b)),
                "command" => p.commands.push(v.command(b)),
                _ => p.mcps.push(v.mcp(b)),
            }
        });
        p.description = attrs.str("description");
        p
    }

    fn skill(&mut self, b: &Block) -> Skill {
        let attrs = self.leaf(b, SKILL_ATTRS);
        Skill {
            name: b.label.clone(),
            source: self.require(&attrs, b, "source"),
            description: attrs.str("description"),
        }
    }

    fn command(&mut self, b: &Block) -> Command {
        let attrs = self.leaf(b, COMMAND_ATTRS);
        Command {
            name: b.label.clone(),
            source: self.require(&attrs, b, "source"),
            description: attrs.str("description"),
        }
    }

    fn agent(&mut self, b: &Block) -> Agent {
        let attrs = self.leaf(b, AGENT_ATTRS);
        Agent {
            name: b.label.clone(),
            source: self.require(&attrs, b, "source"),
            description: attrs.str("description"),
            model: attrs.str("model"),
            tools: attrs.list("tools"),
        }
    }

    fn mcp(&mut self, b: &Block) -> Mcp {
        let attrs = self.leaf(b, MCP_ATTRS);
        let raw = self.require(&attrs, b, "transport");
        let transport = Transport::from_name(&raw);
        if let Some(t) = transport {
            let needed = if t == Transport::Stdio { "command" } else { "url" };
            if attrs.get(needed).is_none() {
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
            transport: transport.unwrap_or(Transport::Stdio),
            description: attrs.str("description"),
            command: attrs.str("command"),
            args: attrs.list("args"),
            env: attrs.map("env"),
            url: attrs.str("url"),
            headers: attrs.map("headers"),
        }
    }

    fn leaf<'a>(&mut self, b: &'a Block, allowed: &[(&str, Kind)]) -> Attrs<'a> {
        let ctx = b.kind.clone();
        self.walk(&ctx, &b.body, allowed, |v, child| v.unknown_block(child, &ctx))
    }

    fn require(&mut self, attrs: &Attrs<'_>, b: &Block, name: &str) -> String {
        if attrs.get(name).is_none() {
            self.report(
                IssueKind::MissingRequired,
                b.line,
                b.col,
                format!("missing required attribute \"{name}\" in {} \"{}\"", b.kind, b.label),
            );
        }
        attrs.str(name).unwrap_or_default()
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
