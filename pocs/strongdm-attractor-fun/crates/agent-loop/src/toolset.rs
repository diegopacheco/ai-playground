use crate::tools::{self, Tool};
use llm_client::types::ToolDefinition;

pub struct Toolset {
    tools: Vec<Box<dyn Tool>>,
}

impl Toolset {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }

    pub fn default_set() -> Self {
        Self::new(tools::all_tools())
    }

    pub fn for_provider(provider: &str) -> Self {
        let mut all = tools::all_tools();
        match provider {
            "openai" => {
                all.retain(|t| t.name() != "edit_file");
            }
            "anthropic" => {
                all.retain(|t| t.name() != "apply_patch");
            }
            _ => {}
        }
        Self::new(all)
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    pub async fn execute(&self, name: &str, input: serde_json::Value) -> Result<String, String> {
        let tool = self
            .find(name)
            .ok_or_else(|| format!("unknown tool: {}", name))?;
        tool.execute(input).await
    }
}
