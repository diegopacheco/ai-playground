pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "claude".to_string(),
        vec![
            "-p".to_string(),
            prompt.to_string(),
            "--model".to_string(),
            "sonnet".to_string(),
            "--output-format".to_string(),
            "text".to_string(),
            "--tools".to_string(),
            "".to_string(),
            "--dangerously-skip-permissions".to_string(),
        ],
    )
}
