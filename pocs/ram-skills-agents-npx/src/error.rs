pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(message.into().into())
}
