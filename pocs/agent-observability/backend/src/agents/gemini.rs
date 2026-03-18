pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "gemini".to_string(),
        vec![
            "-y".to_string(),
            "-p".to_string(),
            prompt.to_string(),
        ],
    )
}
