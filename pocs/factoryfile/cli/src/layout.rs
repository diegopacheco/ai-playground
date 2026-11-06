use std::collections::HashMap;

use factoryfile::{Factory, Stage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub route: Option<usize>,
    pub bounded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub layers: Vec<Vec<usize>>,
    pub forward: Vec<Edge>,
    pub backward: Vec<Edge>,
}

pub fn layout(factory: &Factory) -> Layout {
    let index: HashMap<&str, usize> = factory.stages.iter().enumerate().map(|(i, s)| (s.name.as_str(), i)).collect();
    let edges: Vec<Vec<Edge>> = factory.stages.iter().enumerate().map(|(i, s)| stage_edges(i, s, &index)).collect();
    let (forward, backward) = split_edges(&edges, entry_first(factory, &index));
    let layer_of = match grid(factory) {
        Some(cells) => cells.iter().map(|(_, row)| *row).collect(),
        None => longest_path(edges.len(), &forward),
    };
    let depth = layer_of.iter().max().map_or(0, |d| d + 1);
    let mut layers = vec![Vec::new(); depth];
    for (stage, layer) in layer_of.iter().enumerate() {
        layers[*layer].push(stage);
    }
    if let Some(cells) = grid(factory) {
        layers.retain(|l| !l.is_empty());
        for layer in &mut layers {
            layer.sort_by_key(|s| cells[*s].0);
        }
    }
    Layout { layers, forward, backward }
}

pub fn grid(factory: &Factory) -> Option<Vec<(usize, usize)>> {
    if factory.stages.is_empty() || factory.stages.iter().any(|s| s.at.len() != 2) {
        return None;
    }
    Some(factory.stages.iter().map(|s| (s.at[0] as usize, s.at[1] as usize)).collect())
}

fn stage_edges(from: usize, stage: &Stage, index: &HashMap<&str, usize>) -> Vec<Edge> {
    let next = stage.next.iter().map(|n| (n, None, false));
    let routes = stage.routes.iter().enumerate().map(|(i, r)| (&r.to, Some(i), r.cycle || r.exhausted.is_some()));
    next.chain(routes)
        .filter_map(|(to, route, bounded)| Some(Edge { from, to: *index.get(to.as_str())?, route, bounded }))
        .collect()
}

fn entry_first(factory: &Factory, index: &HashMap<&str, usize>) -> Vec<usize> {
    let entry = factory.entry.as_deref().and_then(|e| index.get(e).copied());
    entry.into_iter().chain(0..factory.stages.len()).collect()
}

fn split_edges(edges: &[Vec<Edge>], roots: Vec<usize>) -> (Vec<Edge>, Vec<Edge>) {
    let mut on_stack = vec![None; edges.len()];
    let mut forward = Vec::new();
    let mut backward = Vec::new();
    for root in roots {
        if on_stack[root].is_none() {
            visit(root, edges, &mut on_stack, &mut forward, &mut backward);
        }
    }
    (forward, backward)
}

fn visit(stage: usize, edges: &[Vec<Edge>], on_stack: &mut [Option<bool>], forward: &mut Vec<Edge>, backward: &mut Vec<Edge>) {
    on_stack[stage] = Some(true);
    for edge in &edges[stage] {
        match on_stack[edge.to] {
            Some(true) => backward.push(*edge),
            Some(false) => forward.push(*edge),
            None => {
                forward.push(*edge);
                visit(edge.to, edges, on_stack, forward, backward);
            }
        }
    }
    on_stack[stage] = Some(false);
}

fn longest_path(count: usize, forward: &[Edge]) -> Vec<usize> {
    let mut layer = vec![0; count];
    let mut incoming = vec![0; count];
    for e in forward {
        incoming[e.to] += 1;
    }
    let mut ready: Vec<usize> = (0..count).filter(|s| incoming[*s] == 0).collect();
    while let Some(stage) = ready.pop() {
        for e in forward.iter().filter(|e| e.from == stage) {
            layer[e.to] = layer[e.to].max(layer[stage] + 1);
            incoming[e.to] -= 1;
            if incoming[e.to] == 0 {
                ready.push(e.to);
            }
        }
    }
    layer
}

pub fn route_text(stage: &Stage, edge: &Edge) -> Option<String> {
    let route = &stage.routes[edge.route?];
    Some(match &route.label {
        Some(label) => format!("{} · {label}", route.outcome),
        None => route.outcome.clone(),
    })
}

pub fn work(stage: &Stage) -> String {
    match (&stage.run, &stage.script) {
        (Some(run), Some(script)) => format!("run {run} {script}"),
        (Some(run), None) => format!("run {run}"),
        _ if stage.agents.is_empty() => "no agents".to_string(),
        _ => stage.agents.join(", "),
    }
}

pub fn count(n: usize, word: &str) -> String {
    if n == 1 { format!("1 {word}") } else { format!("{n} {word}s") }
}
