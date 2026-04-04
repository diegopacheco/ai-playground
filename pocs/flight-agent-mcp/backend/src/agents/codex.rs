pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "exec".to_string(),
            "--full-auto".to_string(),
            prompt.to_string(),
        ],
    )
}
