use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Llm,
    Tool,
    Human,
    Condition,
    ParallelStart,
    ParallelEnd,
    Transform,
    Subgraph,
}

impl NodeType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "llm" => NodeType::Llm,
            "tool" => NodeType::Tool,
            "human" => NodeType::Human,
            "condition" => NodeType::Condition,
            "parallel_start" => NodeType::ParallelStart,
            "parallel_end" => NodeType::ParallelEnd,
            "transform" => NodeType::Transform,
            "subgraph" => NodeType::Subgraph,
            _ => NodeType::Llm,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineNode {
    pub id: String,
    pub node_type: NodeType,
    pub attrs: HashMap<String, String>,
    pub successors: Vec<String>,
    pub predecessors: Vec<String>,
}

impl PipelineNode {
    pub fn new(id: String, attrs: HashMap<String, String>) -> Self {
        let node_type = attrs
            .get("type")
            .map(|s| NodeType::from_str(s))
            .unwrap_or(NodeType::Llm);
        Self {
            id,
            node_type,
            attrs,
            successors: Vec::new(),
            predecessors: Vec::new(),
        }
    }

    pub fn prompt(&self) -> Option<&str> {
        self.attrs.get("prompt").map(|s| s.as_str())
    }

    pub fn model(&self) -> Option<&str> {
        self.attrs.get("model").map(|s| s.as_str())
    }

    pub fn operation(&self) -> Option<&str> {
        self.attrs.get("operation").map(|s| s.as_str())
    }

    pub fn tool_name(&self) -> Option<&str> {
        self.attrs.get("tool").map(|s| s.as_str())
    }

    pub fn condition_expr(&self) -> Option<&str> {
        self.attrs.get("condition").map(|s| s.as_str())
    }

    pub fn true_branch(&self) -> Option<&str> {
        self.attrs.get("true_branch").map(|s| s.as_str())
    }

    pub fn false_branch(&self) -> Option<&str> {
        self.attrs.get("false_branch").map(|s| s.as_str())
    }
}
