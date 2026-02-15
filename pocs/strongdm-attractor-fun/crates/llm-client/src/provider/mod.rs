pub mod openai;
pub mod anthropic;
pub mod gemini;

use crate::error::LlmError;
use crate::types::{Request, Response};

pub enum Provider {
    OpenAi(openai::OpenAiProvider),
    Anthropic(anthropic::AnthropicProvider),
    Gemini(gemini::GeminiProvider),
}

impl Provider {
    pub async fn complete(&self, request: &Request) -> Result<Response, LlmError> {
        match self {
            Provider::OpenAi(p) => p.complete(request).await,
            Provider::Anthropic(p) => p.complete(request).await,
            Provider::Gemini(p) => p.complete(request).await,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Provider::OpenAi(_) => "openai",
            Provider::Anthropic(_) => "anthropic",
            Provider::Gemini(_) => "gemini",
        }
    }
}
