pub fn build_command() -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "-m".to_string(),
            "o4-mini".to_string(),
        ],
    )
}
