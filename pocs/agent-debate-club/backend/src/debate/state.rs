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
    pub style_a: String,
    pub style_b: String,
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
        style_a: String,
        style_b: String,
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
            style_a,
            style_b,
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

    pub fn current_style(&self) -> &str {
        match self.current_agent {
            CurrentAgent::A => &self.style_a,
            CurrentAgent::B => &self.style_b,
        }
    }

    pub fn add_message(&mut self, msg: MessageRecord) {
        self.messages.push(msg);
        self.current_agent = self.current_agent.switch();
    }

    fn get_style_prompt(&self) -> String {
        let style = self.current_style();
        match style {
            "ArthurSchopenhauer" => "\n\nAdopt the style of Arthur Schopenhauer: use rhetorical tricks, sophistry, and clever stratagems to win the argument regardless of truth. Be cynical, pessimistic, and use sharp wit.".to_string(),
            "ExtremeRadical" => "\n\nBe extremely radical and provocative. Take the most extreme position possible. Use dramatic language, make bold claims, and be uncompromising in your views.".to_string(),
            "Zen" => "\n\nRespond in a Zen Buddhist style: be calm, philosophical, use paradoxes and koans. Speak with serene wisdom and detachment. Question the nature of the debate itself.".to_string(),
            "Idiocracy" => "\n\nRespond like a character from Idiocracy: use simple words, be easily distracted, make nonsensical arguments, and reference consumer products and entertainment. Be confidently wrong.".to_string(),
            "comedian" => "\n\nRespond like a stand-up comedian: use humor, jokes, punchlines, and comedic timing. Make fun of the topic and your opponent. Be witty and entertaining above all else.".to_string(),
            "gangster" => "\n\nRespond like a gangster from a crime movie: use street slang, be tough and intimidating, make veiled threats, and speak with swagger and bravado. Demand respect.".to_string(),
            "political_candidate" => "\n\nRespond like a political candidate: dodge direct questions, pivot to talking points, make vague promises, attack your opponent's character, and appeal to emotions over logic. Never admit being wrong.".to_string(),
            _ => "".to_string(),
        }
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

        let style_instruction = self.get_style_prompt();

        format!(
            r#"You are Agent {} in a debate about: "{}"

Previous messages:
{}

Respond with your argument. Start with [ATTACK] or [DEFENSE]:
- ATTACK: Challenge the opposing view
- DEFENSE: Defend your position

Keep response to 2-3 sentences. Be persuasive.{}"#,
            self.current_agent_label(),
            self.topic,
            history,
            style_instruction
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
