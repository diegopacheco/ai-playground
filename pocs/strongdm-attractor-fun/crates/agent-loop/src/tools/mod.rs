pub mod read_file;
pub mod write_file;
pub mod edit_file;
pub mod shell;
pub mod grep;
pub mod glob;
pub mod apply_patch;
pub mod spawn_agent;

use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

pub type ToolResult = Result<String, String>;
pub type ToolFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    fn execute(&self, input: Value) -> ToolFuture;
}

pub fn all_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(read_file::ReadFileTool),
        Box::new(write_file::WriteFileTool),
        Box::new(edit_file::EditFileTool),
        Box::new(shell::ShellTool),
        Box::new(grep::GrepTool),
        Box::new(glob::GlobTool),
        Box::new(apply_patch::ApplyPatchTool),
        Box::new(spawn_agent::SpawnAgentTool::new()),
    ]
}
