use std::collections::{BTreeMap, HashSet};

use crate::error::Error;
use crate::model::{Agent, Exhausted, Factory, Factoryfile, Loop, Route, Stage};
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
    UnknownReference,
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
const FACTORY_ATTRS: &[(&str, Kind)] = &[
    ("title", Kind::Str),
    ("description", Kind::Str),
    ("entry", Kind::Str),
    ("flow", Kind::Str),
    ("backlog", Kind::Str),
];
const AGENT_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("role", Kind::Str),
    ("provider", Kind::Str),
    ("model", Kind::Str),
    ("tools", Kind::List),
    ("source", Kind::Str),
    ("content", Kind::Str),
];
const STAGE_ATTRS: &[(&str, Kind)] = &[
    ("label", Kind::Str),
    ("description", Kind::Str),
    ("agents", Kind::List),
    ("run", Kind::Str),
    ("script", Kind::Str),
    ("next", Kind::List),
    ("artifacts", Kind::List),
    ("output", Kind::Str),
    ("check", Kind::Str),
    ("verdict", Kind::Map),
    ("source", Kind::Str),
    ("content", Kind::Str),
    ("skills", Kind::List),
    ("tasks", Kind::Str),
    ("parallel", Kind::Str),
    ("visits", Kind::Str),
    ("end", Kind::Str),
    ("at", Kind::List),
];
const ON_ATTRS: &[(&str, Kind)] = &[
    ("to", Kind::Str),
    ("label", Kind::Str),
    ("reason", Kind::Str),
    ("cycle", Kind::Str),
];
const EXHAUSTED_ATTRS: &[(&str, Kind)] = &[("label", Kind::Str), ("reason", Kind::Str)];
const VERDICT_KEYS: &[&str] = &["field", "yes", "no", "list", "notes"];
const LOOP_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("from", Kind::Str),
    ("to", Kind::Str),
    ("until", Kind::Str),
    ("max", Kind::Str),
];
const FACTORY_KINDS: &[&str] = &["agent", "stage", "loop"];
const REFERENCES: &[(&str, &str, &str)] = &[
    ("factory", "entry", "stage"),
    ("stage", "agents", "agent"),
    ("stage", "next", "stage"),
    ("stage", "tasks", "stage"),
    ("on", "to", "stage"),
    ("loop", "from", "stage"),
    ("loop", "to", "stage"),
];

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

    fn map(&self, name: &str) -> Option<BTreeMap<String, String>> {
        match self.value(name) {
            Some(Value::Map(entries)) => Some(entries.clone()),
            _ => None,
        }
    }

    fn number(&self, name: &str) -> Option<u32> {
        self.str(name).and_then(|s| s.parse().ok())
    }
}

struct Reference {
    target: &'static str,
    names: Vec<String>,
    owner: String,
    line: usize,
    col: usize,
}

pub fn is_valid_max(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some('1'..='9')) && value.len() <= 6 && chars.all(|c| c.is_ascii_digit())
}

pub fn is_valid_at(value: &str) -> bool {
    value == "0" || ((1..=3).contains(&value.len()) && !value.starts_with('0') && value.chars().all(|c| c.is_ascii_digit()))
}

pub fn validate(statements: &[Statement]) -> (Factoryfile, Vec<Issue>) {
    let mut v = Validator {
        issues: Vec::new(),
        references: Vec::new(),
    };
    let file = v.file(statements);
    (file, v.issues)
}

struct Validator {
    issues: Vec<Issue>,
    references: Vec<Reference>,
}

impl Validator {
    fn report(&mut self, kind: IssueKind, line: usize, col: usize, message: String) {
        self.issues.push(Issue {
            kind,
            error: Error::new(line, col, message),
        });
    }

    fn file(&mut self, statements: &[Statement]) -> Factoryfile {
        let mut factories: Vec<Factory> = Vec::new();
        let mut seen = HashSet::new();
        let attrs = self.walk("file", "file", statements, FILE_ATTRS, |v, b| {
            if b.kind != "factory" {
                v.unknown_block(b, "file");
                return;
            }
            if !seen.insert(b.label.clone()) {
                v.report(IssueKind::DuplicateName, b.line, b.col, format!("duplicate factory \"{}\"", b.label));
            }
            factories.push(v.factory(b));
        });
        if !attrs.has("version") {
            self.report(IssueKind::MissingRequired, 1, 1, "missing required attribute \"version\"".to_string());
        }
        Factoryfile {
            version: attrs.str("version").unwrap_or_default(),
            factories,
        }
    }

    fn factory(&mut self, block: &Block) -> Factory {
        self.references.clear();
        let mut factory = Factory {
            name: block.label.clone(),
            ..Factory::default()
        };
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let owner = format!("factory \"{}\"", block.label);
        let attrs = self.walk("factory", &owner, &block.body, FACTORY_ATTRS, |v, b| {
            if !FACTORY_KINDS.contains(&b.kind.as_str()) {
                v.unknown_block(b, "factory");
                return;
            }
            if !seen.insert((b.kind.clone(), b.label.clone())) {
                v.report(
                    IssueKind::DuplicateName,
                    b.line,
                    b.col,
                    format!("duplicate {} \"{}\" in factory \"{}\"", b.kind, b.label, block.label),
                );
            }
            match b.kind.as_str() {
                "agent" => factory.agents.push(v.agent(b)),
                "stage" => factory.stages.push(v.stage(b)),
                _ => factory.loops.push(v.looping(b)),
            }
        });
        self.check_references(&seen);
        factory.title = attrs.str("title");
        factory.description = attrs.str("description");
        factory.flow = attrs.str("flow");
        factory.backlog = attrs.str("backlog");
        factory.entry = attrs.str("entry").or_else(|| factory.stages.first().map(|s| s.name.clone()));
        factory
    }

    fn check_references(&mut self, defined: &HashSet<(String, String)>) {
        let references = std::mem::take(&mut self.references);
        for r in references {
            for name in r.names.iter().filter(|n| !defined.contains(&(r.target.to_string(), (*n).clone()))) {
                self.report(
                    IssueKind::UnknownReference,
                    r.line,
                    r.col,
                    format!("unknown {} \"{name}\" in {}", r.target, r.owner),
                );
            }
        }
    }

    fn agent(&mut self, b: &Block) -> Agent {
        let attrs = self.leaf(b, AGENT_ATTRS);
        let (has_source, has_content) = (attrs.has("source"), attrs.has("content"));
        if !has_source && !has_content {
            self.missing(b, "\"source\" or \"content\"");
        } else if has_source && has_content {
            self.report(
                IssueKind::Conflict,
                b.line,
                b.col,
                format!("agent \"{}\" cannot set both \"source\" and \"content\"", b.label),
            );
        }
        Agent {
            name: b.label.clone(),
            description: attrs.str("description"),
            role: attrs.str("role"),
            provider: attrs.str("provider"),
            model: attrs.str("model"),
            tools: attrs.list("tools"),
            source: attrs.str("source"),
            content: attrs.str("content"),
        }
    }

    fn stage(&mut self, b: &Block) -> Stage {
        let owner = format!("stage \"{}\"", b.label);
        let mut routes: Vec<Route> = Vec::new();
        let attrs = self.walk("stage", &owner, &b.body, STAGE_ATTRS, |v, child| {
            if child.kind != "on" {
                v.unknown_block(child, "stage");
                return;
            }
            let duplicated = routes.iter().any(|r| r.outcome == child.label);
            if duplicated {
                v.report(
                    IssueKind::DuplicateName,
                    child.line,
                    child.col,
                    format!("duplicate on \"{}\" in stage \"{}\"", child.label, b.label),
                );
            }
            let route = v.route(child, &owner);
            if !duplicated {
                routes.push(route);
            }
        });
        if !attrs.has("agents") && !attrs.has("run") {
            self.missing(b, "\"agents\" or \"run\"");
        }
        if attrs.has("source") && attrs.has("content") {
            self.conflict(b, "cannot set both \"source\" and \"content\"".to_string());
        }
        let on_next = routes.iter().any(|r| r.outcome == "next");
        if attrs.has("next") && on_next {
            self.conflict(b, "cannot set both \"next\" and on \"next\"".to_string());
        }
        if let Some(r) = routes.iter().find(|r| r.outcome != "next").filter(|_| attrs.has("next") || on_next) {
            self.conflict(b, format!("cannot mix the next route with on \"{}\"", r.outcome));
        }
        Stage {
            name: b.label.clone(),
            label: attrs.str("label"),
            description: attrs.str("description"),
            agents: attrs.list("agents"),
            run: attrs.str("run"),
            script: attrs.str("script"),
            next: attrs.list("next"),
            artifacts: attrs.list("artifacts"),
            output: attrs.str("output"),
            check: attrs.str("check"),
            verdict: attrs.map("verdict"),
            source: attrs.str("source"),
            content: attrs.str("content"),
            skills: attrs.list("skills"),
            tasks: attrs.str("tasks"),
            parallel: attrs.number("parallel"),
            visits: attrs.number("visits"),
            end: attrs.str("end"),
            at: attrs.list("at").iter().filter_map(|n| n.parse().ok()).collect(),
            routes,
        }
    }

    fn route(&mut self, b: &Block, owner: &str) -> Route {
        let mut exhausted = None;
        let attrs = self.walk("on", owner, &b.body, ON_ATTRS, |v, child| v.fallback(b, child, owner, &mut exhausted));
        if !attrs.has("to") {
            self.missing(b, "\"to\"");
        }
        Route {
            outcome: b.label.clone(),
            to: attrs.str("to").unwrap_or_default(),
            label: attrs.str("label"),
            reason: attrs.str("reason"),
            cycle: attrs.str("cycle").as_deref() == Some("true"),
            exhausted,
        }
    }

    fn fallback(&mut self, parent: &Block, b: &Block, owner: &str, slot: &mut Option<Box<Exhausted>>) {
        if b.kind != "exhausted" {
            self.unknown_block(b, &parent.kind);
            return;
        }
        if slot.is_some() {
            self.report(
                IssueKind::DuplicateName,
                b.line,
                b.col,
                format!("duplicate exhausted in {} \"{}\"", parent.kind, parent.label),
            );
        }
        self.references.push(Reference {
            target: "stage",
            names: vec![b.label.clone()],
            owner: owner.to_string(),
            line: b.line,
            col: b.col,
        });
        let mut next = None;
        let attrs = self.walk("exhausted", owner, &b.body, EXHAUSTED_ATTRS, |v, child| v.fallback(b, child, owner, &mut next));
        if slot.is_none() {
            *slot = Some(Box::new(Exhausted {
                to: b.label.clone(),
                label: attrs.str("label"),
                reason: attrs.str("reason"),
                exhausted: next,
            }));
        }
    }

    fn looping(&mut self, b: &Block) -> Loop {
        let attrs = self.leaf(b, LOOP_ATTRS);
        for required in ["from", "to"] {
            if !attrs.has(required) {
                self.missing(b, &format!("\"{required}\""));
            }
        }
        Loop {
            name: b.label.clone(),
            description: attrs.str("description"),
            from: attrs.str("from").unwrap_or_default(),
            to: attrs.str("to").unwrap_or_default(),
            until: attrs.str("until"),
            max: attrs.str("max").and_then(|m| m.parse().ok()),
        }
    }

    fn missing(&mut self, b: &Block, what: &str) {
        self.report(
            IssueKind::MissingRequired,
            b.line,
            b.col,
            format!("{} \"{}\" requires {what}", b.kind, b.label),
        );
    }

    fn conflict(&mut self, b: &Block, what: String) {
        self.report(IssueKind::Conflict, b.line, b.col, format!("{} \"{}\" {what}", b.kind, b.label));
    }

    fn leaf<'a>(&mut self, b: &'a Block, allowed: &[(&str, Kind)]) -> Attrs<'a> {
        let owner = format!("{} \"{}\"", b.kind, b.label);
        self.walk(&b.kind, &owner, &b.body, allowed, |v, child| v.unknown_block(child, &b.kind))
    }

    fn unknown_block(&mut self, b: &Block, ctx: &str) {
        self.report(IssueKind::UnknownBlock, b.line, b.col, format!("unknown block \"{}\" in {ctx}", b.kind));
    }

    fn walk<'a>(
        &mut self,
        ctx: &str,
        owner: &str,
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
            self.collect_reference(ctx, owner, a);
            accepted.push(a);
        }
        Attrs(accepted)
    }

    fn check_value(&mut self, ctx: &str, a: &Attribute) {
        let issue = match (ctx, a.name.as_str(), &a.value) {
            ("file", "version", Value::Str(s)) if s != "1" => (IssueKind::InvalidValue, format!("unsupported version \"{s}\"")),
            ("loop", "max", Value::Str(s)) | ("stage", "parallel" | "visits", Value::Str(s)) if !is_valid_max(s) => (
                IssueKind::InvalidValue,
                format!("invalid {} \"{s}\", expected an integer from 1 to 999999", a.name),
            ),
            ("stage", "at", Value::List(items)) if items.len() != 2 || !items.iter().all(|i| is_valid_at(i)) => {
                (IssueKind::InvalidValue, "invalid at, expected two integers from 0 to 999".to_string())
            }
            ("on", "cycle", Value::Str(s)) if s != "true" && s != "false" => {
                (IssueKind::InvalidValue, format!("invalid cycle \"{s}\", expected \"true\" or \"false\""))
            }
            ("stage", "verdict", Value::Map(entries)) => match entries.keys().find(|k| !VERDICT_KEYS.contains(&k.as_str())) {
                Some(key) => (IssueKind::UnknownAttribute, format!("unknown key \"{key}\" in verdict")),
                None if !entries.contains_key("field") => (IssueKind::MissingRequired, "verdict requires \"field\"".to_string()),
                None => return,
            },
            _ => return,
        };
        self.report(issue.0, a.value_line, a.value_col, issue.1);
    }

    fn collect_reference(&mut self, ctx: &str, owner: &str, a: &Attribute) {
        let Some((_, _, target)) = REFERENCES.iter().find(|(c, n, _)| *c == ctx && *n == a.name) else {
            return;
        };
        let names = match &a.value {
            Value::Str(s) => vec![s.clone()],
            Value::List(items) => items.clone(),
            Value::Map(_) => return,
        };
        self.references.push(Reference {
            target,
            names,
            owner: owner.to_string(),
            line: a.value_line,
            col: a.value_col,
        });
    }
}

fn matches_kind(value: &Value, kind: Kind) -> bool {
    matches!((value, kind), (Value::Str(_), Kind::Str) | (Value::List(_), Kind::List) | (Value::Map(_), Kind::Map))
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Str => "a string",
        Kind::List => "a list",
        Kind::Map => "a map",
    }
}
