use crate::types::StreamEvent;

pub fn parse_sse_line(line: &str) -> Option<String> {
    if line.starts_with("data: ") {
        let data = &line[6..];
        if data == "[DONE]" {
            return None;
        }
        Some(data.to_string())
    } else {
        None
    }
}

pub fn parse_sse_buffer(buffer: &str) -> Vec<String> {
    buffer
        .lines()
        .filter_map(parse_sse_line)
        .collect()
}

pub struct SseParser {
    buffer: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        while let Some(pos) = self.buffer.find("\n\n") {
            let block = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();
            for line in block.lines() {
                if let Some(data) = parse_sse_line(line) {
                    events.push(data);
                }
            }
        }
        if self.buffer.contains('\n') && !self.buffer.contains("\n\n") {
            let lines: Vec<&str> = self.buffer.lines().collect();
            if lines.len() > 1 {
                let last = lines.last().unwrap().to_string();
                for line in &lines[..lines.len() - 1] {
                    if let Some(data) = parse_sse_line(line) {
                        events.push(data);
                    }
                }
                self.buffer = last;
            }
        }
        events
    }
}

pub type StreamCallback = Box<dyn Fn(StreamEvent) + Send>;
