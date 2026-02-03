pub fn build_command() -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "-m".to_string(),
            "gpt-5.2-codex".to_string(),
        ],
    )
}
