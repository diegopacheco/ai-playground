pub mod runner;
pub mod claude;
pub mod codex;
pub mod copilot;
pub mod gemini;

use runner::JudgeRunner;

pub fn get_runner(judge_name: &str) -> JudgeRunner {
    JudgeRunner::new(judge_name)
}

pub fn get_available_judges() -> Vec<&'static str> {
    vec!["claude", "codex", "copilot", "gemini"]
}
