use super::runner::AgentRunner;

pub async fn call_claude(prompt: &str) -> Result<String, String> {
    let runner = AgentRunner::new(
        "claude".to_string(),
        vec![
            "-p".to_string(),
            prompt.to_string(),
            "--model".to_string(),
            "sonnet".to_string(),
            "--dangerously-skip-permissions".to_string(),
        ],
    );
    runner.run().await
}
