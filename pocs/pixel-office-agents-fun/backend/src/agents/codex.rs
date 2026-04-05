pub fn build(prompt: &str) -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "-p".to_string(),
            prompt.to_string(),
        ],
    )
}
