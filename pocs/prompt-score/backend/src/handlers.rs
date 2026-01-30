use axum::{
    extract::Json,
    http::StatusCode,
    response::sse::{Event, Sse},
};
use futures::stream::Stream;
use futures::StreamExt;
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use crate::models::{AnalyzeRequest, ProgressEvent, ModelResult};
use crate::score_engine::calculate_scores;
use crate::agents::{claude, codex, copilot, gemini};

pub async fn analyze_stream(
    Json(payload): Json<AnalyzeRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    if payload.prompt.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let (tx, rx) = mpsc::channel::<ProgressEvent>(16);
    let prompt = payload.prompt.clone();

    tokio::spawn(async move {
        let _ = tx.send(ProgressEvent::Start {
            total_steps: 6,
            message: "Starting analysis...".to_string(),
        }).await;

        let scores = calculate_scores(&prompt).await;
        let _ = tx.send(ProgressEvent::ScoresReady {
            scores: scores.clone(),
            step: 1,
            message: "Dimension scores calculated by claude/opus-4.5".to_string(),
        }).await;

        let agents = [
            "claude/opus-4.5",
            "codex/gpt-5.2-codex",
            "copilot/sonnet4",
            "gemini/gemini-3",
        ];

        let mut model_results: Vec<ModelResult> = Vec::new();

        for (i, agent_name) in agents.iter().enumerate() {
            let step = (i + 2) as u8;
            let _ = tx.send(ProgressEvent::AgentStart {
                agent: agent_name.to_string(),
                step,
                message: format!("Querying {}...", agent_name),
            }).await;

            let result = match i {
                0 => claude::analyze(&prompt).await,
                1 => codex::analyze(&prompt).await,
                2 => copilot::analyze(&prompt).await,
                _ => gemini::analyze(&prompt).await,
            };

            let _ = tx.send(ProgressEvent::AgentDone {
                agent: agent_name.to_string(),
                result: result.clone(),
                step,
                message: format!("{} completed", agent_name),
            }).await;

            model_results.push(result);
        }

        let _ = tx.send(ProgressEvent::Complete {
            scores,
            model_results,
            message: "Analysis complete!".to_string(),
        }).await;
    });

    let stream = ReceiverStream::new(rx).map(|event| {
        let data = serde_json::to_string(&event).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    Ok(Sse::new(stream))
}

pub async fn health() -> &'static str {
    "OK"
}
