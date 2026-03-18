use axum::extract::{Path, State};
use axum::response::sse::{Event, Sse};
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::agents;
use crate::otel::{
    SpanRecord, TraceRecord, build_step_prompt, estimate_tokens,
    export_to_jaeger, generate_span_id, generate_trace_id, now_ns,
};
use crate::AppState;

#[derive(Deserialize)]
pub struct RunRequest {
    pub topic: String,
    pub agent: String,
}

#[derive(Serialize)]
pub struct RunResponse {
    pub trace_id: String,
    pub status: String,
}

pub async fn run_agent(
    State(state): State<AppState>,
    Json(payload): Json<RunRequest>,
) -> Json<RunResponse> {
    let trace_id = generate_trace_id();
    let root_span_id = generate_span_id();
    let now = chrono::Utc::now().to_rfc3339();
    let start_ns = now_ns();

    let trace = TraceRecord {
        trace_id: trace_id.clone(),
        topic: payload.topic.clone(),
        agent: payload.agent.clone(),
        started_at: now,
        finished_at: None,
        spans: vec![SpanRecord {
            trace_id: trace_id.clone(),
            span_id: root_span_id.clone(),
            parent_span_id: None,
            name: "agent-run".to_string(),
            start_time_ns: start_ns,
            end_time_ns: None,
            duration_ms: None,
            attributes: HashMap::from([
                ("agent.type".to_string(), payload.agent.clone()),
                ("agent.topic".to_string(), payload.topic.clone()),
            ]),
            status: "running".to_string(),
        }],
        result: None,
        total_tokens_estimated: 0,
        total_duration_ms: 0,
        status: "running".to_string(),
    };

    {
        let mut traces = state.traces.lock().unwrap();
        traces.push(trace);
    }

    let tid = trace_id.clone();
    let topic = payload.topic.clone();
    let agent = payload.agent.clone();
    let traces = state.traces.clone();
    let broadcaster = state.broadcaster.clone();

    tokio::spawn(async move {
        let steps = ["analyze", "plan", "research", "synthesize"];
        let mut context = topic.clone();
        let mut total_tokens: u64 = 0;

        for step in steps {
            let span_id = generate_span_id();
            let step_start_ns = now_ns();

            broadcaster.send(&serde_json::json!({
                "type": "step_started",
                "trace_id": tid,
                "step": step,
                "span_id": span_id,
            }).to_string());

            let prompt = build_step_prompt(step, &topic, &context);
            let input_tokens = estimate_tokens(&prompt);
            let runner = agents::get_runner(&agent);
            let result = runner.run(&prompt).await;

            let step_end_ns = now_ns();
            let duration_ms = (step_end_ns - step_start_ns) / 1_000_000;

            let (output, status) = match &result {
                Ok(text) => (text.clone(), "ok".to_string()),
                Err(e) => (e.clone(), "error".to_string()),
            };

            let output_tokens = estimate_tokens(&output);
            total_tokens += input_tokens + output_tokens;

            let span = SpanRecord {
                trace_id: tid.clone(),
                span_id: span_id.clone(),
                parent_span_id: Some(root_span_id.clone()),
                name: step.to_string(),
                start_time_ns: step_start_ns,
                end_time_ns: Some(step_end_ns),
                duration_ms: Some(duration_ms),
                attributes: HashMap::from([
                    ("step.name".to_string(), step.to_string()),
                    ("agent.type".to_string(), agent.clone()),
                    ("tokens.input".to_string(), input_tokens.to_string()),
                    ("tokens.output".to_string(), output_tokens.to_string()),
                    ("tokens.total".to_string(), (input_tokens + output_tokens).to_string()),
                    ("output.length".to_string(), output.len().to_string()),
                    ("output.preview".to_string(), output.chars().take(200).collect::<String>()),
                ]),
                status: status.clone(),
            };

            {
                let mut all_traces = traces.lock().unwrap();
                if let Some(t) = all_traces.iter_mut().find(|t| t.trace_id == tid) {
                    t.spans.push(span.clone());
                }
            }

            broadcaster.send(&serde_json::json!({
                "type": "step_completed",
                "trace_id": tid,
                "span": span,
            }).to_string());

            if status == "ok" {
                context = output;
            }
        }

        let end_ns = now_ns();
        let total_duration_ms = (end_ns - start_ns) / 1_000_000;

        let trace_for_export = {
            let mut all_traces = traces.lock().unwrap();
            if let Some(t) = all_traces.iter_mut().find(|t| t.trace_id == tid) {
                t.finished_at = Some(chrono::Utc::now().to_rfc3339());
                t.result = Some(context.clone());
                t.total_tokens_estimated = total_tokens;
                t.total_duration_ms = total_duration_ms;
                t.status = "completed".to_string();

                if let Some(root) = t.spans.iter_mut().find(|s| s.name == "agent-run") {
                    root.end_time_ns = Some(end_ns);
                    root.duration_ms = Some(total_duration_ms);
                    root.status = "ok".to_string();
                    root.attributes.insert("tokens.total".to_string(), total_tokens.to_string());
                }

                Some(t.clone())
            } else {
                None
            }
        };

        if let Some(t) = trace_for_export {
            export_to_jaeger(&t).await;
        }

        broadcaster.send(&serde_json::json!({
            "type": "trace_completed",
            "trace_id": tid,
            "total_duration_ms": total_duration_ms,
            "total_tokens": total_tokens,
        }).to_string());
    });

    Json(RunResponse {
        trace_id,
        status: "started".to_string(),
    })
}

pub async fn get_traces(State(state): State<AppState>) -> Json<Vec<TraceRecord>> {
    let traces = state.traces.lock().unwrap();
    let summaries: Vec<TraceRecord> = traces.iter().rev().cloned().collect();
    Json(summaries)
}

pub async fn get_trace(
    State(state): State<AppState>,
    Path(trace_id): Path<String>,
) -> Json<Option<TraceRecord>> {
    let traces = state.traces.lock().unwrap();
    let trace = traces.iter().find(|t| t.trace_id == trace_id).cloned();
    Json(trace)
}

pub async fn trace_stream(
    State(state): State<AppState>,
    Path(trace_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.broadcaster.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(move |msg| {
            match msg {
                Ok(data) => {
                    if data.contains(&trace_id) {
                        Some(Ok(Event::default().data(data)))
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        });
    Sse::new(stream)
}

pub async fn get_agents() -> Json<Vec<(String, String)>> {
    Json(agents::get_available_agents())
}
