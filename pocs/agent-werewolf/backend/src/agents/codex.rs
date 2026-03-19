pub fn build_command(model: &str, prompt: &str) -> (String, Vec<String>) {
    (
        "codex".to_string(),
        vec![
            "exec".to_string(),
            "--full-auto".to_string(),
            "-m".to_string(),
            model.to_string(),
            prompt.to_string(),
        ],
    )
}
