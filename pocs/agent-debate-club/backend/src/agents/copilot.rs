pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "copilot".to_string(),
        vec![
            "--allow-all".to_string(),
            "--model".to_string(),
            "claude-sonnet-4".to_string(),
            "-p".to_string(),
            prompt.to_string(),
        ],
    )
}
