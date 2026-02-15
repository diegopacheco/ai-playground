use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DotGraph {
    pub name: String,
    pub is_digraph: bool,
    pub nodes: Vec<DotNode>,
    pub edges: Vec<DotEdge>,
    pub subgraphs: Vec<DotSubgraph>,
}

#[derive(Debug, Clone)]
pub struct DotNode {
    pub id: String,
    pub attrs: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct DotEdge {
    pub from: String,
    pub to: String,
    pub attrs: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct DotSubgraph {
    pub name: String,
    pub nodes: Vec<DotNode>,
    pub edges: Vec<DotEdge>,
}

impl DotGraph {
    pub fn new(name: String, is_digraph: bool) -> Self {
        Self {
            name,
            is_digraph,
            nodes: Vec::new(),
            edges: Vec::new(),
            subgraphs: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_graph_new() {
        let g = DotGraph::new("test".into(), true);
        assert_eq!(g.name, "test");
        assert!(g.is_digraph);
        assert!(g.nodes.is_empty());
    }
}
