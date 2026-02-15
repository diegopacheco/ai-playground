use super::{Tool, ToolFuture};
use serde_json::Value;
use std::sync::atomic::{AtomicU32, Ordering};

static DEPTH: AtomicU32 = AtomicU32::new(0);
const MAX_DEPTH: u32 = 3;

pub struct SpawnAgentTool {
    _private: (),
}

impl SpawnAgentTool {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Spawn a sub-agent to handle a subtask"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": { "type": "string", "description": "Task description for the sub-agent" },
                "context": { "type": "string", "description": "Additional context" }
            },
            "required": ["task"]
        })
    }

    fn execute(&self, input: Value) -> ToolFuture {
        Box::pin(async move {
            let current = DEPTH.load(Ordering::SeqCst);
            if current >= MAX_DEPTH {
                return Err(format!("max agent depth {} reached", MAX_DEPTH));
            }
            let task = input["task"]
                .as_str()
                .ok_or_else(|| "missing 'task' parameter".to_string())?;
            DEPTH.fetch_add(1, Ordering::SeqCst);
            let result = format!("sub-agent spawned for task: {}", task);
            DEPTH.fetch_sub(1, Ordering::SeqCst);
            Ok(result)
        })
    }
}

pub fn current_depth() -> u32 {
    DEPTH.load(Ordering::SeqCst)
}
