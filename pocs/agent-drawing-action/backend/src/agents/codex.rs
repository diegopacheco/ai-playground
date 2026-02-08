pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "exec".to_string(),
            "--full-auto".to_string(),
            "--model".to_string(),
            "gpt-5.2-codex".to_string(),
            prompt.to_string(),
        ],
    )
}
