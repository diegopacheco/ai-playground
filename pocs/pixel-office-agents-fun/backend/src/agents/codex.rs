pub fn build(prompt: &str) -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "exec".to_string(),
            "-c".to_string(),
            "model=\"gpt-5.4\"".to_string(),
            prompt.to_string(),
        ],
    )
}
