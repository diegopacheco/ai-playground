use std::collections::VecDeque;

pub struct SteeringQueue {
    messages: VecDeque<String>,
}

impl SteeringQueue {
    pub fn new() -> Self {
        Self {
            messages: VecDeque::new(),
        }
    }

    pub fn push(&mut self, message: String) {
        self.messages.push_back(message);
    }

    pub fn pop(&mut self) -> Option<String> {
        self.messages.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }
}
