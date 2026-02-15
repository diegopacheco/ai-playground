use serde_json::Value;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    LlmRequest { model: String, message_count: usize },
    LlmResponse { text_len: usize, tool_call_count: usize },
    ToolCall { name: String, input: Value },
    ToolResult { name: String, success: bool, output_len: usize },
    LoopDetected,
    MaxTurnsReached { turns: u32 },
    Complete { text: String },
    Error { message: String },
}

pub type EventCallback = Box<dyn Fn(&AgentEvent) + Send + Sync>;

pub struct EventEmitter {
    callbacks: Vec<EventCallback>,
}

impl EventEmitter {
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    pub fn on_event(&mut self, callback: EventCallback) {
        self.callbacks.push(callback);
    }

    pub fn emit(&self, event: &AgentEvent) {
        for cb in &self.callbacks {
            cb(event);
        }
    }
}
