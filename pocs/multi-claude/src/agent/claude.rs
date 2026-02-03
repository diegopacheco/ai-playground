pub fn build_command() -> (String, Vec<String>) {
    (
        "claude".to_string(),
        vec![
            "--model".to_string(),
            "opus-4.5".to_string(),
        ],
    )
}
