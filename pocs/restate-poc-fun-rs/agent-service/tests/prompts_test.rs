use agent_service::prompts::{PromptParams, load_prompt, render_prompt};

const TEMPLATE: &str = "---\nname: t\nparams: $goal, $goal_owner\n---\nOwner $goal_owner wants $goal.\n";

fn params<'a>(pairs: &[(&'a str, &str)]) -> PromptParams<'a> {
    pairs.iter().map(|(name, value)| (*name, value.to_string())).collect()
}

#[test]
fn replaces_every_declared_parameter_longest_name_first_so_goal_does_not_eat_goal_owner() {
    assert_eq!(render_prompt(TEMPLATE, &params(&[("goal", "a trip"), ("goal_owner", "Ana")])).unwrap(), "Owner Ana wants a trip.");
}

#[test]
fn fails_loud_when_a_declared_parameter_has_no_value_so_the_agent_never_sees_a_raw_placeholder() {
    assert!(render_prompt(TEMPLATE, &params(&[("goal", "a trip")])).unwrap_err().0.contains("$goal_owner"));
    assert!(render_prompt(TEMPLATE, &params(&[("goal", "a trip"), ("goal_owner", "  ")])).unwrap_err().0.contains("$goal_owner"));
}

#[test]
fn rejects_a_prompt_file_without_the_header_that_documents_its_parameters() {
    assert!(render_prompt("Do $goal", &params(&[("goal", "x")])).unwrap_err().0.contains("header"));
}

#[test]
fn every_shipped_prompt_renders_with_the_parameters_the_workflow_passes() {
    assert!(load_prompt("plan-goal", &params(&[("goal", "learn rust")])).unwrap().contains("learn rust"));
    assert!(load_prompt("execute-task", &params(&[("goal", "g"), ("task", "t1"), ("previous_work", "none yet")])).unwrap().contains("t1"));
    assert!(load_prompt("summarize-run", &params(&[("goal", "g"), ("work", "w1")])).unwrap().contains("w1"));
}
