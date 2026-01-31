use std::sync::Arc;
use std::time::Instant;
use chrono::Utc;
use sqlx::{Pool, Sqlite};
use crate::agents::get_runner;
use crate::debate::state::DebateState;
use crate::persistence::db;
use crate::persistence::models::MessageRecord;
use crate::sse::broadcaster::{Broadcaster, DebateEvent};

pub struct DebateEngine {
    pool: Pool<Sqlite>,
    broadcaster: Arc<Broadcaster>,
}

impl DebateEngine {
    pub fn new(pool: Pool<Sqlite>, broadcaster: Arc<Broadcaster>) -> Self {
        Self { pool, broadcaster }
    }

    pub async fn run_debate(&self, mut state: DebateState) {
        let start = Instant::now();
        let duration_ms = (state.duration_seconds * 1000) as u64;

        while start.elapsed().as_millis() < duration_ms as u128 {
            let agent_name = state.current_agent_name().to_string();
            let agent_label = state.current_agent_label().to_string();

            self.broadcaster
                .broadcast(
                    &state.debate_id,
                    DebateEvent::AgentThinking {
                        agent: agent_label.clone(),
                    },
                )
                .await;

            let prompt = state.build_debater_prompt();
            let runner = get_runner(&agent_name);

            match runner.run(&prompt).await {
                Ok(response) => {
                    let (stance, content) = parse_response(&response);
                    let msg = MessageRecord {
                        id: 0,
                        debate_id: state.debate_id.clone(),
                        agent: agent_label.clone(),
                        content: content.clone(),
                        stance: stance.clone(),
                        created_at: Utc::now().to_rfc3339(),
                    };

                    if let Err(e) = db::save_message(&self.pool, &msg).await {
                        self.broadcaster
                            .broadcast(
                                &state.debate_id,
                                DebateEvent::Error {
                                    message: format!("Failed to save message: {}", e),
                                },
                            )
                            .await;
                    }

                    self.broadcaster
                        .broadcast(
                            &state.debate_id,
                            DebateEvent::AgentMessage {
                                agent: agent_label,
                                content,
                                stance,
                            },
                        )
                        .await;

                    state.add_message(msg);
                }
                Err(e) => {
                    self.broadcaster
                        .broadcast(
                            &state.debate_id,
                            DebateEvent::Error {
                                message: format!("Agent {} error: {}", agent_name, e),
                            },
                        )
                        .await;
                    break;
                }
            }

            if start.elapsed().as_millis() >= duration_ms as u128 {
                break;
            }
        }

        let total_duration = start.elapsed().as_millis() as u64;
        self.run_judge(&state, total_duration).await;
    }

    async fn run_judge(&self, state: &DebateState, duration_ms: u64) {
        let judge_prompt = state.build_judge_prompt();
        let runner = get_runner(&state.agent_judge);

        match runner.run(&judge_prompt).await {
            Ok(response) => {
                let (winner, reason) = parse_judge_response(&response);

                if let Err(e) = db::update_debate_result(
                    &self.pool,
                    &state.debate_id,
                    &winner,
                    &reason,
                    &Utc::now().to_rfc3339(),
                )
                .await
                {
                    self.broadcaster
                        .broadcast(
                            &state.debate_id,
                            DebateEvent::Error {
                                message: format!("Failed to update result: {}", e),
                            },
                        )
                        .await;
                }

                self.broadcaster
                    .broadcast(
                        &state.debate_id,
                        DebateEvent::DebateOver {
                            winner,
                            reason,
                            duration_ms,
                        },
                    )
                    .await;
            }
            Err(e) => {
                self.broadcaster
                    .broadcast(
                        &state.debate_id,
                        DebateEvent::Error {
                            message: format!("Judge error: {}", e),
                        },
                    )
                    .await;
            }
        }
    }
}

fn parse_response(response: &str) -> (String, String) {
    let trimmed = response.trim();
    if trimmed.starts_with("[ATTACK]") {
        ("ATTACK".to_string(), trimmed.replace("[ATTACK]", "").trim().to_string())
    } else if trimmed.starts_with("[DEFENSE]") {
        ("DEFENSE".to_string(), trimmed.replace("[DEFENSE]", "").trim().to_string())
    } else {
        ("ATTACK".to_string(), trimmed.to_string())
    }
}

fn parse_judge_response(response: &str) -> (String, String) {
    let mut winner = "A".to_string();
    let mut reason = response.trim().to_string();

    for line in response.lines() {
        let line = line.trim();
        if line.starts_with("WINNER:") {
            let w = line.replace("WINNER:", "").trim().to_string();
            if w.contains("B") || w.contains("b") {
                winner = "B".to_string();
            } else {
                winner = "A".to_string();
            }
        } else if line.starts_with("REASON:") {
            reason = line.replace("REASON:", "").trim().to_string();
        }
    }

    (winner, reason)
}
