use crate::agent_sdk::json::Json;
use crate::cache::Entry;
use crate::http::{Request, Response};
use crate::json_out::{list, object, optional_score, optional_text, quote, score};
use crate::service::{Answer, QaService};

pub fn handle(service: &QaService, request: &Request) -> Response {
    let result = match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/api/health") => Ok(object(&[("status", quote("UP"))])),
        ("POST", "/api/ask") => ask(service, &request.body),
        ("GET", "/api/cache") => service.entries().map(|entries| object(&[("entries", list(&entries.iter().map(entry_json).collect::<Vec<_>>()))])),
        ("DELETE", "/api/cache") => service.clear().map(|_| object(&[("cleared", "true".into())])),
        ("GET", "/api/stats") => stats(service),
        _ => return Response::json(404, error("not found")),
    };
    match result {
        Ok(body) => Response::json(200, body),
        Err(message) if message.contains("required") => Response::json(400, error(&message)),
        Err(message) => Response::json(502, error(&message)),
    }
}

fn ask(service: &QaService, body: &str) -> Result<String, String> {
    let json = Json::parse(body).unwrap_or(Json::Null);
    let question = json.get("question").and_then(Json::text).ok_or("question is required")?;
    service.ask(question).map(|answer| answer_json(&answer))
}

fn stats(service: &QaService) -> Result<String, String> {
    let stats = service.stats()?;
    let settings = &service.settings;
    Ok(object(&[
        ("hits", stats.hits.to_string()),
        ("misses", stats.misses.to_string()),
        ("entries", stats.entries.to_string()),
        ("threshold", score(settings.threshold)),
        ("claudeModel", quote(&settings.claude_model)),
        ("embedModel", quote(&settings.embed_model)),
    ]))
}

fn answer_json(answer: &Answer) -> String {
    object(&[
        ("question", quote(&answer.question)),
        ("answer", quote(&answer.answer)),
        ("cached", answer.cached.to_string()),
        ("similarity", optional_score(answer.similarity)),
        ("matchedQuestion", optional_text(answer.matched_question.as_deref())),
        ("latencyMs", answer.latency_ms.to_string()),
    ])
}

fn entry_json(entry: &Entry) -> String {
    object(&[
        ("key", quote(&entry.key)),
        ("question", quote(&entry.question)),
        ("answer", quote(&entry.answer)),
        ("hits", entry.hits.to_string()),
        ("createdAt", entry.created_at.to_string()),
    ])
}

fn error(message: &str) -> String {
    object(&[("error", quote(message))])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::fakes::service;

    fn call(service: &QaService, method: &str, path: &str, body: &str) -> (u16, Json) {
        let response = handle(service, &Request { method: method.into(), path: path.into(), body: body.into() });
        (response.status, Json::object(&response.body))
    }

    #[test]
    fn the_ui_sees_a_miss_then_a_hit_with_the_matched_question() {
        let service = service(false);
        let (_, first) = call(&service, "POST", "/api/ask", r#"{"question":"What is Rust?"}"#);
        let (status, second) = call(&service, "POST", "/api/ask", r#"{"question":"what is rust"}"#);
        assert_eq!(status, 200);
        assert_eq!(first.get("cached"), Some(&Json::Bool(false)));
        assert_eq!(second.get("cached"), Some(&Json::Bool(true)));
        assert_eq!(second.get("matchedQuestion").and_then(Json::text), Some("What is Rust?"));
        let (_, stats) = call(&service, "GET", "/api/stats", "");
        assert_eq!(stats.get("hits").and_then(Json::number), Some(1.0));
        assert_eq!(stats.get("threshold").and_then(Json::number), Some(0.9));
    }

    #[test]
    fn a_malformed_body_is_a_client_error_not_an_upstream_failure() {
        assert_eq!(call(&service(false), "POST", "/api/ask", "nope").0, 400);
    }

    #[test]
    fn a_claude_failure_surfaces_as_bad_gateway_because_the_proxy_upstream_failed() {
        let (status, body) = call(&service(true), "POST", "/api/ask", r#"{"question":"What is Rust?"}"#);
        assert_eq!(status, 502);
        assert_eq!(body.get("error").and_then(Json::text), Some("claude is down"));
    }

    #[test]
    fn the_cache_tab_lists_and_clears_entries() {
        let service = service(false);
        call(&service, "POST", "/api/ask", r#"{"question":"What is Rust?"}"#);
        let (_, listed) = call(&service, "GET", "/api/cache", "");
        assert_eq!(listed.get("entries").and_then(Json::list).map(|entries| entries.len()), Some(1));
        call(&service, "DELETE", "/api/cache", "");
        let (_, listed) = call(&service, "GET", "/api/cache", "");
        assert_eq!(listed.get("entries").and_then(Json::list).map(|entries| entries.len()), Some(0));
    }

    #[test]
    fn unknown_routes_are_not_found() {
        assert_eq!(call(&service(false), "GET", "/nope", "").0, 404);
    }
}
