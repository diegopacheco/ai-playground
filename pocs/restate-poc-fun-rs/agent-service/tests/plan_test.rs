use agent_service::plan::{MAX_TASKS, parse_plan};

#[test]
fn reads_the_task_list_even_when_the_model_wraps_it_in_prose_or_a_code_fence() {
    assert_eq!(parse_plan("Sure!\n```json\n[\"book flight\", \"find hotel\"]\n```").unwrap(), vec!["book flight", "find hotel"]);
}

#[test]
fn caps_the_plan_so_a_chatty_model_cannot_make_the_workflow_run_forever() {
    let many = serde_json::to_string(&(0..10).map(|i| format!("task {i}")).collect::<Vec<_>>()).unwrap();
    assert_eq!(parse_plan(&many).unwrap().len(), MAX_TASKS);
}

#[test]
fn drops_blank_and_non_text_items_instead_of_running_empty_tasks() {
    assert_eq!(parse_plan(r#"["  a  ", "", 3, null, "b"]"#).unwrap(), vec!["a", "b"]);
}

#[test]
fn fails_when_there_is_no_usable_plan_so_restate_retries_the_plan_step() {
    assert!(parse_plan("I cannot help").unwrap_err().0.contains("no JSON task list"));
    assert!(parse_plan("[]").unwrap_err().0.contains("no tasks"));
    assert!(parse_plan("] then [").unwrap_err().0.contains("no JSON task list"));
}
