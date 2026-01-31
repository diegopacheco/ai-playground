use crate::persistence::models::MessageRecord;

pub struct DebateState {
    pub debate_id: String,
    pub topic: String,
    pub agent_a: String,
    pub agent_b: String,
    pub agent_judge: String,
    pub duration_seconds: i64,
    pub messages: Vec<MessageRecord>,
    pub current_agent: CurrentAgent,
}

#[derive(Clone, Copy, PartialEq)]
pub enum CurrentAgent {
    A,
    B,
}

impl CurrentAgent {
    pub fn switch(&self) -> Self {
        match self {
            CurrentAgent::A => CurrentAgent::B,
            CurrentAgent::B => CurrentAgent::A,
        }
    }
}

impl DebateState {
    pub fn new(
        debate_id: String,
        topic: String,
        agent_a: String,
        agent_b: String,
        agent_judge: String,
        duration_seconds: i64,
    ) -> Self {
        Self {
            debate_id,
            topic,
            agent_a,
            agent_b,
            agent_judge,
            duration_seconds,
            messages: Vec::new(),
            current_agent: CurrentAgent::A,
        }
    }

    pub fn current_agent_name(&self) -> &str {
        match self.current_agent {
            CurrentAgent::A => &self.agent_a,
            CurrentAgent::B => &self.agent_b,
        }
    }

    pub fn current_agent_label(&self) -> &str {
        match self.current_agent {
            CurrentAgent::A => "A",
            CurrentAgent::B => "B",
        }
    }

    pub fn add_message(&mut self, msg: MessageRecord) {
        self.messages.push(msg);
        self.current_agent = self.current_agent.switch();
    }

    pub fn build_debater_prompt(&self) -> String {
        let history = if self.messages.is_empty() {
            "None yet - you speak first".to_string()
        } else {
            self.messages
                .iter()
                .map(|m| format!("Agent {}: [{}] {}", m.agent, m.stance, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!(
            r#"You are Agent {} in a debate about: "{}"

Previous messages:
{}

Respond with your argument. Start with [ATTACK] or [DEFENSE]:
- ATTACK: Challenge the opposing view
- DEFENSE: Defend your position

Keep response to 2-3 sentences. Be persuasive."#,
            self.current_agent_label(),
            self.topic,
            history
        )
    }

    pub fn build_judge_prompt(&self) -> String {
        let transcript = self
            .messages
            .iter()
            .map(|m| format!("Agent {}: [{}] {}", m.agent, m.stance, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"You are judging a debate about: "{}"

Transcript:
{}

Declare winner. Consider: argument strength, logic, rebuttals, persuasiveness.

Format:
WINNER: A or B
REASON: Your explanation (2-3 sentences)"#,
            self.topic, transcript
        )
    }
}
