use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use factoryfile::{Attribute, Block, Statement, Value};

use crate::diagnostic::Diagnostic;

const ITEM_KINDS: [&str; 3] = ["agent", "stage", "loop"];

pub fn check(statements: &[Statement], base: &Path) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for factory in blocks(statements).filter(|b| b.kind == "factory") {
        check_factory(factory, base, &mut out);
    }
    out
}

fn check_factory(factory: &Block, base: &Path, out: &mut Vec<Diagnostic>) {
    check_name(factory, out);
    let items: Vec<&Block> = blocks(&factory.body).filter(|b| ITEM_KINDS.contains(&b.kind.as_str())).collect();
    let stages: Vec<&Block> = items.iter().copied().filter(|b| b.kind == "stage").collect();
    if stages.is_empty() {
        out.push(Diagnostic::warning(
            "W001",
            factory.line,
            factory.col,
            format!("factory \"{}\" declares no stage", factory.label),
        ));
    }
    for item in &items {
        check_name(item, out);
        check_content(item, out);
        match item.kind.as_str() {
            "agent" => check_source(item, base, out),
            "loop" => check_max(item, out),
            _ => {}
        }
    }
    check_unused_agents(&items, &stages, out);
    let graph = Graph::new(factory, &stages, &items);
    check_reachable(&graph, &stages, out);
    check_cycles(&graph, &stages, out);
}

fn blocks(statements: &[Statement]) -> impl Iterator<Item = &Block> {
    statements.iter().filter_map(|s| match s {
        Statement::Block(b) => Some(b),
        Statement::Attribute(_) => None,
    })
}

fn attribute<'a>(block: &'a Block, name: &str) -> Option<&'a Attribute> {
    block.body.iter().find_map(|s| match s {
        Statement::Attribute(a) if a.name == name => Some(a),
        _ => None,
    })
}

fn string_attr<'a>(block: &'a Block, name: &str) -> Option<(&'a Attribute, &'a str)> {
    attribute(block, name).and_then(|a| match &a.value {
        Value::Str(s) => Some((a, s.as_str())),
        _ => None,
    })
}

fn list_attr<'a>(block: &'a Block, name: &str) -> Option<(&'a Attribute, &'a [String])> {
    attribute(block, name).and_then(|a| match &a.value {
        Value::List(items) => Some((a, items.as_slice())),
        _ => None,
    })
}

pub fn is_kebab_case(name: &str) -> bool {
    !name.is_empty()
        && name.split('-').all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()))
}

pub fn is_local_path(source: &str) -> bool {
    source.starts_with('.') || source.starts_with('/')
}

fn check_name(block: &Block, out: &mut Vec<Diagnostic>) {
    if !is_kebab_case(&block.label) {
        out.push(Diagnostic::warning(
            "W004",
            block.line,
            block.col,
            format!("{} name \"{}\" is not kebab-case", block.kind, block.label),
        ));
    }
}

fn check_source(block: &Block, base: &Path, out: &mut Vec<Diagnostic>) {
    let Some((attr, source)) = string_attr(block, "source") else { return };
    if !is_local_path(source) || base.join(source).exists() {
        return;
    }
    out.push(Diagnostic::warning(
        "W002",
        attr.value_line,
        attr.value_col,
        format!("source \"{source}\" does not exist"),
    ));
}

fn check_content(block: &Block, out: &mut Vec<Diagnostic>) {
    let Some((attr, content)) = string_attr(block, "content") else { return };
    if content.trim().is_empty() {
        out.push(Diagnostic::warning(
            "W006",
            attr.value_line,
            attr.value_col,
            format!("content of {} \"{}\" is empty", block.kind, block.label),
        ));
    }
}

fn check_max(block: &Block, out: &mut Vec<Diagnostic>) {
    if attribute(block, "max").is_none() {
        out.push(Diagnostic::warning(
            "W008",
            block.line,
            block.col,
            format!("loop \"{}\" has no \"max\", it may never end", block.label),
        ));
    }
}

fn check_unused_agents(items: &[&Block], stages: &[&Block], out: &mut Vec<Diagnostic>) {
    let used: HashSet<&str> = stages
        .iter()
        .filter_map(|s| list_attr(s, "agents"))
        .flat_map(|(_, names)| names.iter().map(String::as_str))
        .collect();
    for agent in items.iter().filter(|b| b.kind == "agent" && !used.contains(b.label.as_str())) {
        out.push(Diagnostic::warning(
            "W005",
            agent.line,
            agent.col,
            format!("agent \"{}\" is not used by any stage", agent.label),
        ));
    }
}

struct Graph<'a> {
    entry: Option<&'a str>,
    next: HashMap<&'a str, Vec<&'a str>>,
    loops: HashMap<&'a str, Vec<&'a str>>,
}

impl<'a> Graph<'a> {
    fn new(factory: &'a Block, stages: &[&'a Block], items: &[&'a Block]) -> Self {
        let known: HashSet<&str> = stages.iter().map(|s| s.label.as_str()).collect();
        let mut next: HashMap<&str, Vec<&str>> = HashMap::new();
        for stage in stages {
            let targets = list_attr(stage, "next").map(|(_, n)| n).unwrap_or_default();
            next.entry(stage.label.as_str())
                .or_default()
                .extend(targets.iter().map(String::as_str).filter(|t| known.contains(t)));
        }
        let mut loops: HashMap<&str, Vec<&str>> = HashMap::new();
        for l in items.iter().filter(|b| b.kind == "loop") {
            if let (Some((_, from)), Some((_, to))) = (string_attr(l, "from"), string_attr(l, "to")) {
                if known.contains(from) && known.contains(to) {
                    loops.entry(from).or_default().push(to);
                }
            }
        }
        let entry = match string_attr(factory, "entry") {
            Some((_, e)) => Some(e).filter(|e| known.contains(e)),
            None => stages.first().map(|s| s.label.as_str()),
        };
        Graph { entry, next, loops }
    }

    fn successors(&self, stage: &str) -> impl Iterator<Item = &'a str> + '_ {
        let next = self.next.get(stage).into_iter().flatten();
        let loops = self.loops.get(stage).into_iter().flatten();
        next.chain(loops).copied()
    }
}

fn check_reachable(graph: &Graph<'_>, stages: &[&Block], out: &mut Vec<Diagnostic>) {
    let Some(entry) = graph.entry else { return };
    let mut seen: HashSet<&str> = HashSet::from([entry]);
    let mut queue = VecDeque::from([entry]);
    while let Some(stage) = queue.pop_front() {
        for target in graph.successors(stage) {
            if seen.insert(target) {
                queue.push_back(target);
            }
        }
    }
    for stage in stages.iter().filter(|s| !seen.contains(s.label.as_str())) {
        out.push(Diagnostic::warning(
            "W003",
            stage.line,
            stage.col,
            format!("stage \"{}\" cannot be reached from the entry \"{entry}\"", stage.label),
        ));
    }
}

fn check_cycles(graph: &Graph<'_>, stages: &[&Block], out: &mut Vec<Diagnostic>) {
    let by_name: HashMap<&str, &Block> = stages.iter().map(|s| (s.label.as_str(), *s)).collect();
    let mut state: HashMap<&str, bool> = HashMap::new();
    let roots = graph.entry.into_iter().chain(stages.iter().map(|s| s.label.as_str()));
    for root in roots {
        if !state.contains_key(root) {
            visit(root, graph, &by_name, &mut state, out);
        }
    }
}

fn visit<'a>(
    stage: &'a str,
    graph: &Graph<'a>,
    by_name: &HashMap<&str, &Block>,
    state: &mut HashMap<&'a str, bool>,
    out: &mut Vec<Diagnostic>,
) {
    state.insert(stage, true);
    for target in graph.next.get(stage).into_iter().flatten().copied() {
        match state.get(target) {
            Some(true) => {
                let Some((attr, _)) = by_name.get(stage).and_then(|b| list_attr(b, "next")) else { continue };
                out.push(Diagnostic::warning(
                    "W007",
                    attr.value_line,
                    attr.value_col,
                    format!("stage \"{stage}\" goes back to \"{target}\" through \"next\", use a loop with a max"),
                ));
            }
            Some(false) => {}
            None => visit(target, graph, by_name, state, out),
        }
    }
    state.insert(stage, false);
}
