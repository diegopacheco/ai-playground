use crate::agent_sdk::json::Json;
use crate::http::post_json;
use crate::json_out::{object, quote};

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
}

pub struct OllamaEmbedder {
    addr: String,
    model: String,
}

impl OllamaEmbedder {
    pub fn new(addr: &str, model: &str) -> Self {
        Self { addr: addr.to_string(), model: model.to_string() }
    }
}

impl Embedder for OllamaEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let body = object(&[("model", quote(&self.model)), ("input", quote(text))]);
        parse_embedding(&post_json(&self.addr, "/api/embed", &body)?)
    }
}

pub fn parse_embedding(raw: &str) -> Result<Vec<f32>, String> {
    let json = Json::object(raw);
    let values = json
        .get("embeddings")
        .and_then(Json::list)
        .and_then(|embeddings| embeddings.first())
        .and_then(Json::list)
        .filter(|values| !values.is_empty())
        .ok_or_else(|| format!("Ollama returned no embedding: {}", raw.chars().take(200).collect::<String>()))?;
    Ok(values.iter().filter_map(Json::number).map(|value| value as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_embedding_is_used_as_the_question_vector() {
        assert_eq!(parse_embedding(r#"{"model":"m","embeddings":[[0.5,-1,2e-1]]}"#), Ok(vec![0.5, -1.0, 0.2]));
    }

    #[test]
    fn an_ollama_error_fails_loud_instead_of_caching_an_empty_vector() {
        assert!(parse_embedding(r#"{"error":"model not found"}"#).unwrap_err().contains("model not found"));
        assert!(parse_embedding(r#"{"embeddings":[[]]}"#).is_err());
    }
}
