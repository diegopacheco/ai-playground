pub fn build_command(prompt: &str, model: &str) -> (String, Vec<String>) {
    (
        "claude".to_string(),
        vec![
            "-p".to_string(),
            prompt.to_string(),
            "--model".to_string(),
            model.to_string(),
            "--dangerously-skip-permissions".to_string(),
        ],
    )
}
