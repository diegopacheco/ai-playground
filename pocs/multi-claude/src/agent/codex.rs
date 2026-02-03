pub fn build_command() -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "--model".to_string(),
            "gpt-5.2".to_string(),
        ],
    )
}
