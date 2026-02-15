pub mod types;
pub mod error;
pub mod retry;
pub mod catalog;
pub mod stream;
pub mod provider;

use error::LlmError;
use provider::Provider;
use types::{Request, Response};
use retry::{RetryConfig, with_retry};

pub struct LlmClient {
    provider: Provider,
    retry_config: RetryConfig,
}

impl LlmClient {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            retry_config: RetryConfig::default(),
        }
    }

    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    pub fn from_env(provider_name: &str) -> Result<Self, LlmError> {
        let provider = match provider_name {
            "openai" => {
                let key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| LlmError::Auth("OPENAI_API_KEY not set".into()))?;
                Provider::OpenAi(provider::openai::OpenAiProvider::new(key))
            }
            "anthropic" => {
                let key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| LlmError::Auth("ANTHROPIC_API_KEY not set".into()))?;
                Provider::Anthropic(provider::anthropic::AnthropicProvider::new(key))
            }
            "gemini" => {
                let key = std::env::var("GEMINI_API_KEY")
                    .map_err(|_| LlmError::Auth("GEMINI_API_KEY not set".into()))?;
                Provider::Gemini(provider::gemini::GeminiProvider::new(key))
            }
            _ => return Err(LlmError::InvalidRequest(format!("unknown provider: {}", provider_name))),
        };
        Ok(Self::new(provider))
    }

    pub fn from_model(model_id: &str) -> Result<Self, LlmError> {
        let provider_name = catalog::provider_for_model(model_id)
            .ok_or_else(|| LlmError::InvalidRequest(format!("unknown model: {}", model_id)))?;
        Self::from_env(provider_name)
    }

    pub async fn complete(&self, request: &Request) -> Result<Response, LlmError> {
        with_retry(&self.retry_config, || self.provider.complete(request)).await
    }

    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }
}
