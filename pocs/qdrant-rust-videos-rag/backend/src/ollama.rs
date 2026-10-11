use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::config::Config;

#[derive(Clone)]
pub struct Ollama {
    http: reqwest::Client,
    url: String,
    embed_model: String,
    chat_model: String,
}

impl Ollama {
    pub fn new(config: &Config) -> Self {
        Self {
            http: reqwest::Client::new(),
            url: config.ollama_url.clone(),
            embed_model: config.embed_model.clone(),
            chat_model: config.chat_model.clone(),
        }
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}{}", self.url, path))
            .json(&body)
            .send()
            .await
            .with_context(|| format!("ollama {path} unreachable"))?
            .error_for_status()
            .with_context(|| format!("ollama {path} failed"))?;
        Ok(response.json().await?)
    }

    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let body = json!({"model": self.embed_model, "input": inputs});
        let response = self.post("/api/embed", body).await?;
        let vectors: Vec<Vec<f32>> = serde_json::from_value(response["embeddings"].clone())
            .context("ollama embed returned no embeddings")?;
        anyhow::ensure!(
            vectors.len() == inputs.len(),
            "ollama returned {} embeddings for {} inputs",
            vectors.len(),
            inputs.len()
        );
        Ok(vectors)
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed(&[format!("search_query: {text}")]).await?;
        vectors.pop().context("ollama returned no embedding")
    }

    pub async fn chat(&self, prompt: &str) -> Result<String> {
        let body = json!({
            "model": self.chat_model,
            "stream": false,
            "messages": [{"role": "user", "content": prompt}]
        });
        let response = self.post("/api/chat", body).await?;
        response["message"]["content"]
            .as_str()
            .map(|content| content.trim().to_string())
            .context("ollama chat returned no message")
    }
}
