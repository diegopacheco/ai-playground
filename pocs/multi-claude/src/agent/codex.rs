pub fn build_command() -> (String, Vec<String>) {
    (
        "script".to_string(),
        vec![
            "-q".to_string(),
            "/dev/null".to_string(),
            "codex".to_string(),
            "-m".to_string(),
            "gpt-5.2-codex".to_string(),
        ],
    )
}
