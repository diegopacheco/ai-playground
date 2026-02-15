use crate::condition::parse_condition;
use crate::graph::PipelineGraph;
use crate::human::prompt_human;
use crate::node::NodeType;
use crate::state::State;
use crate::stylesheet::Stylesheet;
use crate::transform::apply_transform;
use llm_client::types::{Message, Request};
use llm_client::LlmClient;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Pipeline {
    graph: PipelineGraph,
    stylesheet: Stylesheet,
    client: Option<LlmClient>,
    default_model: String,
}

impl Pipeline {
    pub fn new(
        graph: PipelineGraph,
        stylesheet: Stylesheet,
        client: LlmClient,
        default_model: String,
    ) -> Self {
        Self {
            graph,
            stylesheet,
            client: Some(client),
            default_model,
        }
    }

    pub fn without_client(
        graph: PipelineGraph,
        stylesheet: Stylesheet,
        default_model: String,
    ) -> Self {
        Self {
            graph,
            stylesheet,
            client: None,
            default_model,
        }
    }

    pub async fn run(&self) -> Result<State, String> {
        let state = Arc::new(Mutex::new(State::new()));
        self.execute_ordered(&state).await?;
        let final_state = state.lock().await.clone();
        Ok(final_state)
    }

    async fn execute_ordered(&self, state: &Arc<Mutex<State>>) -> Result<(), String> {
        let mut i = 0;
        while i < self.graph.order.len() {
            let node_id = &self.graph.order[i];
            let node = self
                .graph
                .get(node_id)
                .ok_or_else(|| format!("node {} not found", node_id))?
                .clone();

            match node.node_type {
                NodeType::ParallelStart => {
                    let mut parallel_nodes = Vec::new();
                    i += 1;
                    while i < self.graph.order.len() {
                        let nid = &self.graph.order[i];
                        let n = self.graph.get(nid).unwrap().clone();
                        if n.node_type == NodeType::ParallelEnd {
                            break;
                        }
                        parallel_nodes.push(n);
                        i += 1;
                    }
                    let mut handles = Vec::new();
                    for pn in parallel_nodes {
                        let state_clone = Arc::clone(state);
                        let prompt = {
                            let s = state_clone.lock().await;
                            pn.prompt()
                                .map(|p| s.resolve_template(p))
                                .unwrap_or_default()
                        };
                        let node_id = pn.id.clone();
                        let node_type = pn.node_type.clone();
                        let operation = pn.operation().map(|s| s.to_string());
                        handles.push(tokio::spawn(async move {
                            match node_type {
                                NodeType::Transform => {
                                    if let Some(op) = &operation {
                                        let s = state_clone.lock().await;
                                        let input =
                                            s.get("last_output").unwrap_or("").to_string();
                                        let result =
                                            apply_transform(op, &input, &s)?;
                                        drop(s);
                                        let mut s = state_clone.lock().await;
                                        s.set(&format!("{}_output", node_id), result);
                                    }
                                }
                                _ => {
                                    let mut s = state_clone.lock().await;
                                    s.set(
                                        &format!("{}_output", node_id),
                                        format!("parallel result for {}: {}", node_id, prompt),
                                    );
                                }
                            }
                            Ok::<(), String>(())
                        }));
                    }
                    for handle in handles {
                        handle.await.map_err(|e| format!("join error: {}", e))??;
                    }
                }
                NodeType::ParallelEnd => {}
                NodeType::Condition => {
                    let expr_str = node.condition_expr().unwrap_or("true");
                    let expr = parse_condition(expr_str)?;
                    let s = state.lock().await;
                    let result = expr.evaluate(&s);
                    drop(s);
                    let branch = if result {
                        node.true_branch()
                    } else {
                        node.false_branch()
                    };
                    if let Some(target) = branch {
                        let mut s = state.lock().await;
                        s.set("condition_result", result.to_string());
                        s.set("next_node", target.to_string());
                    }
                }
                NodeType::Llm => {
                    let client = self.client.as_ref()
                        .ok_or_else(|| "LLM client not configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY".to_string())?;
                    let s = state.lock().await;
                    let prompt = node
                        .prompt()
                        .map(|p| s.resolve_template(p))
                        .unwrap_or_else(|| {
                            s.get("last_output").unwrap_or("").to_string()
                        });
                    drop(s);
                    let model = self.resolve_model(&node.id);
                    let config = self.stylesheet.config_for(&node.id);
                    let mut request = Request::new(&model, vec![Message::user(&prompt)]);
                    request.system = config.system_prompt;
                    if let Some(temp) = config.temperature {
                        request.temperature = Some(temp);
                    }
                    if let Some(max) = config.max_tokens {
                        request.max_tokens = max;
                    }
                    let response = client
                        .complete(&request)
                        .await
                        .map_err(|e| e.to_string())?;
                    let text = response.text();
                    let mut s = state.lock().await;
                    s.set(&format!("{}_output", node.id), text.clone());
                    s.set("last_output", text);
                }
                NodeType::Tool => {
                    let tool_name = node.tool_name().unwrap_or("shell");
                    let s = state.lock().await;
                    let input = node
                        .prompt()
                        .map(|p| s.resolve_template(p))
                        .unwrap_or_default();
                    drop(s);
                    let tool_input = serde_json::json!({ "command": input });
                    let toolset = agent_loop::Toolset::default_set();
                    let result = toolset.execute(tool_name, tool_input).await;
                    let output = match result {
                        Ok(out) => out,
                        Err(err) => format!("tool error: {}", err),
                    };
                    let mut s = state.lock().await;
                    s.set(&format!("{}_output", node.id), output.clone());
                    s.set("last_output", output);
                }
                NodeType::Human => {
                    let prompt_text = node.prompt().unwrap_or("Enter input");
                    let s = state.lock().await;
                    let resolved = s.resolve_template(prompt_text);
                    drop(s);
                    let input = prompt_human(&resolved)?;
                    let mut s = state.lock().await;
                    s.set(&format!("{}_output", node.id), input.clone());
                    s.set("last_output", input);
                }
                NodeType::Transform => {
                    let operation = node.operation().unwrap_or("template");
                    let s = state.lock().await;
                    let input = node
                        .prompt()
                        .map(|p| s.resolve_template(p))
                        .unwrap_or_else(|| {
                            s.get("last_output").unwrap_or("").to_string()
                        });
                    let result = apply_transform(operation, &input, &s)?;
                    drop(s);
                    let mut s = state.lock().await;
                    s.set(&format!("{}_output", node.id), result.clone());
                    s.set("last_output", result);
                }
                NodeType::Subgraph => {
                    let mut s = state.lock().await;
                    s.push_scope();
                    s.set(
                        &format!("{}_output", node.id),
                        "subgraph executed".to_string(),
                    );
                    s.pop_scope();
                }
            }
            i += 1;
        }
        Ok(())
    }

    fn resolve_model(&self, node_id: &str) -> String {
        let config = self.stylesheet.config_for(node_id);
        config
            .model
            .or_else(|| {
                self.graph
                    .get(node_id)
                    .and_then(|n| n.model().map(|s| s.to_string()))
            })
            .unwrap_or_else(|| self.default_model.clone())
    }
}
