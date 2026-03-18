use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SpanRecord {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub start_time_ns: u64,
    pub end_time_ns: Option<u64>,
    pub duration_ms: Option<u64>,
    pub attributes: HashMap<String, String>,
    pub status: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TraceRecord {
    pub trace_id: String,
    pub topic: String,
    pub agent: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub spans: Vec<SpanRecord>,
    pub result: Option<String>,
    pub total_tokens_estimated: u64,
    pub total_duration_ms: u64,
    pub status: String,
}

pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub fn generate_trace_id() -> String {
    let mut rng = rand::thread_rng();
    let bytes: [u8; 16] = rng.r#gen();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn generate_span_id() -> String {
    let mut rng = rand::thread_rng();
    let bytes: [u8; 8] = rng.r#gen();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn estimate_tokens(text: &str) -> u64 {
    (text.len() as u64) / 4
}

pub fn build_step_prompt(step: &str, topic: &str, context: &str) -> String {
    match step {
        "analyze" => format!(
            "You are an AI agent performing step 1 of 4: ANALYSIS. \
             Analyze the following topic and identify the key concepts, \
             challenges, and important aspects to address. \
             Topic: {}. Provide a brief analysis in 3-5 sentences.", topic
        ),
        "plan" => format!(
            "You are an AI agent performing step 2 of 4: PLANNING. \
             Based on this analysis: {}. \
             Create a structured plan to address the topic: {}. \
             List 3-5 key points to cover.", context, topic
        ),
        "research" => format!(
            "You are an AI agent performing step 3 of 4: RESEARCH. \
             Following this plan: {}. \
             Provide detailed information about: {}. \
             Be thorough and factual.", context, topic
        ),
        "synthesize" => format!(
            "You are an AI agent performing step 4 of 4: SYNTHESIS. \
             Using this research: {}. \
             Write a clear, comprehensive final answer about: {}. \
             Synthesize all findings into a cohesive response.", context, topic
        ),
        _ => format!("Respond about: {}", topic),
    }
}

pub async fn export_to_jaeger(trace: &TraceRecord) {
    let spans_json: Vec<serde_json::Value> = trace.spans.iter().map(|s| {
        let attrs: Vec<serde_json::Value> = s.attributes.iter().map(|(k, v)| {
            serde_json::json!({
                "key": k,
                "value": {"stringValue": v}
            })
        }).collect();

        serde_json::json!({
            "traceId": s.trace_id,
            "spanId": s.span_id,
            "parentSpanId": s.parent_span_id.clone().unwrap_or_default(),
            "name": s.name,
            "kind": 1,
            "startTimeUnixNano": s.start_time_ns.to_string(),
            "endTimeUnixNano": s.end_time_ns.unwrap_or(now_ns()).to_string(),
            "attributes": attrs,
            "status": {"code": 1}
        })
    }).collect();

    let body = serde_json::json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "agent-observability"}}
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "agent-runner", "version": "0.1.0"},
                "spans": spans_json
            }]
        }]
    });

    let client = reqwest::Client::new();
    match client.post("http://localhost:4318/v1/traces")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(resp) => eprintln!("Exported trace to Jaeger: {}", resp.status()),
        Err(e) => eprintln!("Failed to export to Jaeger: {}", e),
    }
}
