use axum::Json;
use serde::{Deserialize, Serialize};
use crate::agents;

#[derive(Deserialize)]
pub struct SearchRequest {
    pub origin: String,
    pub origin_city: String,
    pub destination: String,
    pub destination_city: String,
    pub date: String,
    pub agent: String,
}

#[derive(Serialize, Clone)]
pub struct FlightResult {
    pub id: String,
    pub date: String,
    pub origin: String,
    pub destination: String,
    pub airline: String,
    pub price: String,
    pub cabin: String,
    pub stops: String,
    pub duration: String,
    pub source: String,
    pub booking_url: String,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<FlightResult>,
    pub agent_used: String,
    pub raw_output: String,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
}

pub async fn get_agents() -> Json<Vec<AgentInfo>> {
    let agents = agents::get_available_agents()
        .into_iter()
        .map(|(id, name)| AgentInfo { id, name })
        .collect();
    Json(agents)
}

pub async fn search_flights(Json(req): Json<SearchRequest>) -> Json<SearchResponse> {
    let prompt = format!(
r#"Search flights from {origin} ({origin_city}) to {dest} ({dest_city}) around {date}.

Use the seats-aero skill and the serpapi skill from the travel-hacking-toolkit to search for award flights and cash prices on this route.

CRITICAL RULES:
- Do NOT complain about missing API keys. Do NOT ask the user to configure anything.
- If a skill or API call fails or keys are missing, skip that source and still return results from whatever works.
- If no API works, return your best knowledge of real airlines and typical prices for this route.
- NEVER return an empty response. Always return flight data.

Your ENTIRE response must be ONLY a JSON array. No text before. No text after. No markdown fences. No explanation. Start with [ end with ].

Each object must have these string fields:
{{"date":"{date}","origin":"{origin}","destination":"{dest}","airline":"United","price":"$850","cabin":"economy","stops":"0","duration":"5h 30m","source":"google_flights","booking_url":""}}

Rules:
- "price" is a string like "$850" or "45000 miles"
- "stops" is a string like "0" or "1"
- "source" is "seats.aero" or "google_flights"
- "booking_url" is a string URL or ""
- Start your response with [ and end with ]"#,
        origin = req.origin,
        origin_city = req.origin_city,
        dest = req.destination,
        dest_city = req.destination_city,
        date = req.date,
    );

    println!("=== Searching: {} -> {} on {} via {} ===", req.origin, req.destination, req.date, req.agent);

    let runner = agents::get_runner(&req.agent);
    match runner.run(&prompt).await {
        Ok(output) => {
            println!("=== Raw output ({} chars) ===\n{}\n=== End ===", output.len(), &output[..output.len().min(2000)]);
            let results = parse_flight_results(&output);
            println!("=== Parsed {} flight results ===", results.len());
            let error = if results.is_empty() {
                Some(format!("Agent returned data but no flights could be parsed. Raw output length: {} chars", output.len()))
            } else {
                None
            };
            Json(SearchResponse {
                results,
                agent_used: req.agent,
                raw_output: output,
                error,
            })
        }
        Err(e) => {
            println!("=== Agent error: {} ===", e);
            Json(SearchResponse {
                results: vec![],
                agent_used: req.agent,
                raw_output: String::new(),
                error: Some(e),
            })
        }
    }
}

fn parse_flight_results(output: &str) -> Vec<FlightResult> {
    let json_str = extract_json_array(output);
    match serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
        Ok(arr) => arr
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| {
                let date = get_str(&v, "date");
                let origin = get_str(&v, "origin");
                let destination = get_str(&v, "destination");
                if date.is_empty() && origin.is_empty() && destination.is_empty() {
                    return None;
                }
                let url = get_str(&v, "booking_url");
                let booking_url = if url.is_empty() {
                    build_fallback_url(&origin, &destination, &date)
                } else {
                    url
                };
                Some(FlightResult {
                    id: format!("flight-{}", i),
                    date,
                    origin,
                    destination,
                    airline: get_str(&v, "airline"),
                    price: get_str_or_number(&v, "price"),
                    cabin: get_str_with_default(&v, "cabin", "economy"),
                    stops: get_str_or_number(&v, "stops"),
                    duration: get_str(&v, "duration"),
                    source: get_str(&v, "source"),
                    booking_url,
                })
            })
            .collect(),
        Err(_) => vec![],
    }
}

fn get_str(v: &serde_json::Value, key: &str) -> String {
    v.get(key)
        .and_then(|val| val.as_str())
        .unwrap_or("")
        .to_string()
}

fn get_str_with_default(v: &serde_json::Value, key: &str, default: &str) -> String {
    let s = get_str(v, key);
    if s.is_empty() { default.to_string() } else { s }
}

fn get_str_or_number(v: &serde_json::Value, key: &str) -> String {
    match v.get(key) {
        Some(val) if val.is_string() => val.as_str().unwrap_or("").to_string(),
        Some(val) if val.is_number() => val.to_string(),
        _ => String::new(),
    }
}

fn build_fallback_url(origin: &str, destination: &str, date: &str) -> String {
    format!(
        "https://www.google.com/travel/flights?q=flights+from+{}+to+{}+on+{}",
        origin, destination, date
    )
}

fn extract_json_array(text: &str) -> String {
    let cleaned = strip_markdown_fences(text);
    if let Some(start) = cleaned.find('[') {
        if let Some(end) = cleaned.rfind(']') {
            if end > start {
                return cleaned[start..=end].to_string();
            }
        }
    }
    "[]".to_string()
}

fn strip_markdown_fences(text: &str) -> String {
    let mut result = String::new();
    let mut inside_fence = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            inside_fence = !inside_fence;
            continue;
        }
        if inside_fence || !trimmed.starts_with("```") {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}
