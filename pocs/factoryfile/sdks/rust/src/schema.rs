use std::collections::HashSet;

use crate::error::Error;
use crate::model::{Agent, Factory, Factoryfile, Loop, Stage};
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
}

const FILE_ATTRS: &[(&str, Kind)] = &[("version", Kind::Str)];
const FACTORY_ATTRS: &[(&str, Kind)] = &[("description", Kind::Str), ("entry", Kind::Str)];
const AGENT_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("role", Kind::Str),
    ("model", Kind::Str),
    ("tools", Kind::List),
    ("source", Kind::Str),
    ("content", Kind::Str),
];
const STAGE_ATTRS: &[(&str, Kind)] = &[
    ("description", Kind::Str),
    ("agents", Kind::List),
    ("next", Kind::List),
    ("artifacts", Kind::List),
    ("content", Kind::Str),
];
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
        let attrs = self.walk("file", "", statements, FILE_ATTRS, |v, b| {
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
        let attrs = self.walk("factory", &block.label, &block.body, FACTORY_ATTRS, |v, b| {
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
        factory.description = attrs.str("description");
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
            model: attrs.str("model"),
            tools: attrs.list("tools"),
            source: attrs.str("source"),
            content: attrs.str("content"),
        }
    }

    fn stage(&mut self, b: &Block) -> Stage {
        let attrs = self.leaf(b, STAGE_ATTRS);
        if !attrs.has("agents") {
            self.missing(b, "\"agents\"");
        }
        Stage {
            name: b.label.clone(),
            description: attrs.str("description"),
            agents: attrs.list("agents"),
            next: attrs.list("next"),
            artifacts: attrs.list("artifacts"),
            content: attrs.str("content"),
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

    fn leaf<'a>(&mut self, b: &'a Block, allowed: &[(&str, Kind)]) -> Attrs<'a> {
        self.walk(&b.kind, &b.label, &b.body, allowed, |v, child| v.unknown_block(child, &b.kind))
    }

    fn unknown_block(&mut self, b: &Block, ctx: &str) {
        self.report(IssueKind::UnknownBlock, b.line, b.col, format!("unknown block \"{}\" in {ctx}", b.kind));
    }

    fn walk<'a>(
        &mut self,
        ctx: &str,
        label: &str,
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
            self.collect_reference(ctx, label, a);
            accepted.push(a);
        }
        Attrs(accepted)
    }

    fn check_value(&mut self, ctx: &str, a: &Attribute) {
        let Value::Str(s) = &a.value else { return };
        let message = match (ctx, a.name.as_str()) {
            ("file", "version") if s != "1" => format!("unsupported version \"{s}\""),
            ("loop", "max") if !is_valid_max(s) => format!("invalid max \"{s}\", expected an integer from 1 to 999999"),
            _ => return,
        };
        self.report(IssueKind::InvalidValue, a.value_line, a.value_col, message);
    }

    fn collect_reference(&mut self, ctx: &str, label: &str, a: &Attribute) {
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
            owner: format!("{ctx} \"{label}\""),
            line: a.value_line,
            col: a.value_col,
        });
    }
}

fn matches_kind(value: &Value, kind: Kind) -> bool {
    matches!((value, kind), (Value::Str(_), Kind::Str) | (Value::List(_), Kind::List))
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Str => "a string",
        Kind::List => "a list",
    }
}
