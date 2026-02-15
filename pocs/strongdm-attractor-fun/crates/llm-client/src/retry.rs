use crate::error::LlmError;
use std::time::Duration;

pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 500,
            max_delay_ms: 30000,
        }
    }
}

impl RetryConfig {
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay = self.base_delay_ms * 2u64.pow(attempt);
        let jitter = (delay as f64 * 0.1) as u64;
        let total = delay.min(self.max_delay_ms) + jitter;
        Duration::from_millis(total)
    }
}

pub async fn with_retry<F, Fut, T>(config: &RetryConfig, mut f: F) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, LlmError>>,
{
    let mut last_err = LlmError::Unknown("no attempts made".to_string());
    for attempt in 0..=config.max_retries {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                if !e.is_retryable() || attempt == config.max_retries {
                    return Err(e);
                }
                let delay = config.delay_for_attempt(attempt);
                tokio::time::sleep(delay).await;
                last_err = e;
            }
        }
    }
    Err(last_err)
}
