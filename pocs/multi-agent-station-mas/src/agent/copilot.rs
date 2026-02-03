pub fn build_command() -> (String, Vec<String>) {
    (
        "copilot".to_string(),
        vec![
            "--model".to_string(),
            "claude-sonnet-4".to_string(),
        ],
    )
}
