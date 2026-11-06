use std::collections::HashMap;

use factoryfile::Factory;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub layers: Vec<Vec<usize>>,
    pub forward: Vec<(usize, usize)>,
    pub backward: Vec<(usize, usize)>,
}

pub fn layout(factory: &Factory) -> Layout {
    let index: HashMap<&str, usize> = factory.stages.iter().enumerate().map(|(i, s)| (s.name.as_str(), i)).collect();
    let edges: Vec<Vec<usize>> = factory
        .stages
        .iter()
        .map(|s| s.next.iter().filter_map(|n| index.get(n.as_str()).copied()).collect())
        .collect();
    let (forward, backward) = split_edges(&edges, entry_first(factory, &index));
    let layer_of = longest_path(edges.len(), &forward);
    let depth = layer_of.iter().max().map_or(0, |d| d + 1);
    let mut layers = vec![Vec::new(); depth];
    for (stage, layer) in layer_of.iter().enumerate() {
        layers[*layer].push(stage);
    }
    Layout { layers, forward, backward }
}

fn entry_first(factory: &Factory, index: &HashMap<&str, usize>) -> Vec<usize> {
    let entry = factory.entry.as_deref().and_then(|e| index.get(e).copied());
    entry.into_iter().chain(0..factory.stages.len()).collect()
}

fn split_edges(edges: &[Vec<usize>], roots: Vec<usize>) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
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

fn visit(
    stage: usize,
    edges: &[Vec<usize>],
    on_stack: &mut [Option<bool>],
    forward: &mut Vec<(usize, usize)>,
    backward: &mut Vec<(usize, usize)>,
) {
    on_stack[stage] = Some(true);
    for &target in &edges[stage] {
        match on_stack[target] {
            Some(true) => backward.push((stage, target)),
            Some(false) => forward.push((stage, target)),
            None => {
                forward.push((stage, target));
                visit(target, edges, on_stack, forward, backward);
            }
        }
    }
    on_stack[stage] = Some(false);
}

fn longest_path(count: usize, forward: &[(usize, usize)]) -> Vec<usize> {
    let mut layer = vec![0; count];
    let mut incoming = vec![0; count];
    for (_, to) in forward {
        incoming[*to] += 1;
    }
    let mut ready: Vec<usize> = (0..count).filter(|s| incoming[*s] == 0).collect();
    while let Some(stage) = ready.pop() {
        for (_, to) in forward.iter().filter(|(from, _)| *from == stage) {
            layer[*to] = layer[*to].max(layer[stage] + 1);
            incoming[*to] -= 1;
            if incoming[*to] == 0 {
                ready.push(*to);
            }
        }
    }
    layer
}

pub fn count(n: usize, word: &str) -> String {
    if n == 1 { format!("1 {word}") } else { format!("{n} {word}s") }
}
