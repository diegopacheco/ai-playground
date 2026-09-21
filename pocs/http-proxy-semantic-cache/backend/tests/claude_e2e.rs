use semantic_cache_proxy::llm::{ClaudeLlm, Llm};

#[test]
fn claude_code_answers_through_the_agent_sdk() {
    let model = std::env::var("CLAUDE_MODEL").unwrap_or_else(|_| "claude-sonnet-5".into());
    let answer = ClaudeLlm::new(&model).answer("What is 2 + 2? Reply with only the number.").unwrap();
    assert!(answer.contains('4'), "unexpected answer: {answer}");
}
