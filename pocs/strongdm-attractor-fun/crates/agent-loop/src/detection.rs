use std::collections::VecDeque;

const WINDOW_SIZE: usize = 10;
const REPEAT_THRESHOLD: usize = 3;

pub struct LoopDetector {
    history: VecDeque<String>,
}

impl LoopDetector {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(WINDOW_SIZE + 1),
        }
    }

    pub fn record(&mut self, tool_name: &str, input_hash: u64) {
        let key = format!("{}:{}", tool_name, input_hash);
        self.history.push_back(key);
        if self.history.len() > WINDOW_SIZE {
            self.history.pop_front();
        }
    }

    pub fn is_looping(&self) -> bool {
        if self.history.len() < REPEAT_THRESHOLD {
            return false;
        }
        let last = self.history.back().unwrap();
        let count = self.history.iter().filter(|h| *h == last).count();
        count >= REPEAT_THRESHOLD
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }
}

pub fn hash_input(input: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let s = input.to_string();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_loop_initially() {
        let det = LoopDetector::new();
        assert!(!det.is_looping());
    }

    #[test]
    fn test_detects_loop() {
        let mut det = LoopDetector::new();
        for _ in 0..3 {
            det.record("read_file", 12345);
        }
        assert!(det.is_looping());
    }

    #[test]
    fn test_varied_calls_no_loop() {
        let mut det = LoopDetector::new();
        det.record("read_file", 1);
        det.record("write_file", 2);
        det.record("shell", 3);
        assert!(!det.is_looping());
    }
}
