use std::fmt;

#[derive(Debug)]
pub enum LlmError {
    Network(String),
    RateLimit(String),
    Auth(String),
    InvalidRequest(String),
    ServerError(String),
    Timeout(String),
    Parse(String),
    Unknown(String),
}

impl LlmError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            LlmError::Network(_) | LlmError::RateLimit(_) | LlmError::ServerError(_) | LlmError::Timeout(_)
        )
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Network(msg) => write!(f, "Network error: {}", msg),
            LlmError::RateLimit(msg) => write!(f, "Rate limit: {}", msg),
            LlmError::Auth(msg) => write!(f, "Auth error: {}", msg),
            LlmError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            LlmError::ServerError(msg) => write!(f, "Server error: {}", msg),
            LlmError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            LlmError::Parse(msg) => write!(f, "Parse error: {}", msg),
            LlmError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl std::error::Error for LlmError {}

impl From<reqwest::Error> for LlmError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            LlmError::Timeout(e.to_string())
        } else if e.is_connect() {
            LlmError::Network(e.to_string())
        } else {
            LlmError::Unknown(e.to_string())
        }
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(e: serde_json::Error) -> Self {
        LlmError::Parse(e.to_string())
    }
}

pub fn classify_status(status: u16, body: &str) -> LlmError {
    match status {
        401 | 403 => LlmError::Auth(body.to_string()),
        429 => LlmError::RateLimit(body.to_string()),
        400 | 422 => LlmError::InvalidRequest(body.to_string()),
        500..=599 => LlmError::ServerError(body.to_string()),
        _ => LlmError::Unknown(format!("status {}: {}", status, body)),
    }
}
