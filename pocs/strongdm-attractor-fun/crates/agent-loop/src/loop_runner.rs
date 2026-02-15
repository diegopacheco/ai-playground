use crate::detection::{hash_input, LoopDetector};
use crate::events::{AgentEvent, EventEmitter};
use crate::session::Session;
use crate::steering::SteeringQueue;
use crate::toolset::Toolset;
use crate::truncation::truncate_output;
use llm_client::types::*;
use llm_client::LlmClient;

pub struct LoopRunner {
    client: LlmClient,
    toolset: Toolset,
    system_prompt: Option<String>,
    max_turns: u32,
    model: String,
}

impl LoopRunner {
    pub fn new(client: LlmClient, toolset: Toolset, model: String) -> Self {
        Self {
            client,
            toolset,
            system_prompt: None,
            max_turns: 50,
            model,
        }
    }

    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    pub fn with_max_turns(mut self, max: u32) -> Self {
        self.max_turns = max;
        self
    }

    pub async fn run(
        &self,
        initial_message: &str,
        events: &EventEmitter,
    ) -> Result<String, String> {
        let mut session = Session::new();
        let mut detector = LoopDetector::new();
        let mut steering = SteeringQueue::new();

        session.add_message(Message::user(initial_message));

        loop {
            if session.turn_count >= self.max_turns {
                events.emit(&AgentEvent::MaxTurnsReached {
                    turns: self.max_turns,
                });
                return Err(format!("max turns {} reached", self.max_turns));
            }

            if let Some(steer_msg) = steering.pop() {
                session.add_message(Message::user(&steer_msg));
            }

            let mut request = Request::new(&self.model, session.messages.clone());
            request.system = self.system_prompt.clone();
            request.tools = self.toolset.definitions();

            events.emit(&AgentEvent::LlmRequest {
                model: self.model.clone(),
                message_count: session.messages.len(),
            });

            let response = self
                .client
                .complete(&request)
                .await
                .map_err(|e| e.to_string())?;

            session.accumulate_usage(&response.usage);
            session.increment_turn();

            events.emit(&AgentEvent::LlmResponse {
                text_len: response.text().len(),
                tool_call_count: response.tool_calls().len(),
            });

            if !response.has_tool_calls() {
                let text = response.text();
                events.emit(&AgentEvent::Complete { text: text.clone() });
                return Ok(text);
            }

            let mut assistant_content = Vec::new();
            let text = response.text();
            if !text.is_empty() {
                assistant_content.push(ContentPart::Text { text });
            }
            for tc in &response.content {
                if matches!(tc, ContentPart::ToolUse { .. }) {
                    assistant_content.push(tc.clone());
                }
            }
            session.add_message(Message {
                role: Role::Assistant,
                content: assistant_content,
            });

            for tc in response.tool_calls() {
                if let ContentPart::ToolUse { id, name, input } = tc {
                    events.emit(&AgentEvent::ToolCall {
                        name: name.clone(),
                        input: input.clone(),
                    });

                    detector.record(name, hash_input(input));
                    if detector.is_looping() {
                        events.emit(&AgentEvent::LoopDetected);
                        steering.push(
                            "You appear to be repeating the same action. Try a different approach."
                                .to_string(),
                        );
                        detector.reset();
                    }

                    let result = self.toolset.execute(name, input.clone()).await;
                    let (success, output) = match result {
                        Ok(out) => (true, truncate_output(&out)),
                        Err(err) => (false, truncate_output(&err)),
                    };

                    events.emit(&AgentEvent::ToolResult {
                        name: name.clone(),
                        success,
                        output_len: output.len(),
                    });

                    session.add_message(Message {
                        role: Role::User,
                        content: vec![ContentPart::ToolResult {
                            tool_use_id: id.clone(),
                            content: output,
                        }],
                    });
                }
            }
        }
    }
}
