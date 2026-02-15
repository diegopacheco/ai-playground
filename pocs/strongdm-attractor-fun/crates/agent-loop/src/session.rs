use llm_client::types::{Message, Usage};

pub struct Session {
    pub messages: Vec<Message>,
    pub total_usage: Usage,
    pub turn_count: u32,
}

impl Session {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            total_usage: Usage::default(),
            turn_count: 0,
        }
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn accumulate_usage(&mut self, usage: &Usage) {
        self.total_usage.input_tokens += usage.input_tokens;
        self.total_usage.output_tokens += usage.output_tokens;
    }

    pub fn increment_turn(&mut self) {
        self.turn_count += 1;
    }
}
