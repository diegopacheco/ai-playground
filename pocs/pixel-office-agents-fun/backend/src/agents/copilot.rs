pub fn build(prompt: &str) -> (String, Vec<String>) {
    (
        "copilot".to_string(),
        vec![
            "-p".to_string(),
            prompt.to_string(),
        ],
    )
}
