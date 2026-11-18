use crate::error::AppError;
use serde_json::Value;

pub const MAX_TASKS: usize = 4;

pub fn parse_plan(output: &str) -> Result<Vec<String>, AppError> {
    let preview: String = output.chars().take(200).collect();
    let (start, end) = match (output.find('['), output.rfind(']')) {
        (Some(start), Some(end)) if end > start => (start, end),
        _ => return Err(AppError::new(format!("agent returned no JSON task list: {preview}"))),
    };
    let values: Vec<Value> = serde_json::from_str(&output[start..=end])
        .map_err(|error| AppError::new(format!("agent plan is not a list: {error}")))?;
    let tasks: Vec<String> = values
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .take(MAX_TASKS)
        .map(String::from)
        .collect();
    if tasks.is_empty() {
        return Err(AppError::new("agent plan has no tasks"));
    }
    Ok(tasks)
}
