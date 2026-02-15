use crate::dot::DotGraph;
use crate::node::PipelineNode;
use std::collections::HashMap;

pub struct PipelineGraph {
    pub nodes: HashMap<String, PipelineNode>,
    pub order: Vec<String>,
}

impl PipelineGraph {
    pub fn from_dot(dot: &DotGraph) -> Result<Self, String> {
        let mut nodes = HashMap::new();

        for dn in &dot.nodes {
            let node = PipelineNode::new(dn.id.clone(), dn.attrs.clone());
            nodes.insert(dn.id.clone(), node);
        }

        for sg in &dot.subgraphs {
            for dn in &sg.nodes {
                let mut attrs = dn.attrs.clone();
                attrs.insert("type".to_string(), "subgraph".to_string());
                attrs.insert("subgraph_name".to_string(), sg.name.clone());
                let node = PipelineNode::new(dn.id.clone(), attrs);
                nodes.insert(dn.id.clone(), node);
            }
        }

        for edge in &dot.edges {
            if !nodes.contains_key(&edge.from) {
                let node = PipelineNode::new(edge.from.clone(), HashMap::new());
                nodes.insert(edge.from.clone(), node);
            }
            if !nodes.contains_key(&edge.to) {
                let node = PipelineNode::new(edge.to.clone(), HashMap::new());
                nodes.insert(edge.to.clone(), node);
            }
            nodes
                .get_mut(&edge.from)
                .unwrap()
                .successors
                .push(edge.to.clone());
            nodes
                .get_mut(&edge.to)
                .unwrap()
                .predecessors
                .push(edge.from.clone());
        }

        let order = topological_sort(&nodes)?;

        Ok(Self { nodes, order })
    }

    pub fn entry_nodes(&self) -> Vec<&str> {
        self.nodes
            .values()
            .filter(|n| n.predecessors.is_empty())
            .map(|n| n.id.as_str())
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<&PipelineNode> {
        self.nodes.get(id)
    }
}

fn topological_sort(nodes: &HashMap<String, PipelineNode>) -> Result<Vec<String>, String> {
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    for (id, node) in nodes {
        in_degree.entry(id.as_str()).or_insert(0);
        for succ in &node.successors {
            *in_degree.entry(succ.as_str()).or_insert(0) += 1;
        }
    }

    let mut queue: Vec<&str> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();
    queue.sort();

    let mut result = Vec::new();
    while let Some(id) = queue.pop() {
        result.push(id.to_string());
        if let Some(node) = nodes.get(id) {
            for succ in &node.successors {
                if let Some(deg) = in_degree.get_mut(succ.as_str()) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(succ.as_str());
                        queue.sort();
                    }
                }
            }
        }
    }

    if result.len() != nodes.len() {
        return Err("cycle detected in graph".to_string());
    }

    Ok(result)
}
