use agent_service::run_state::{Phase, RunState, StepStatus};

fn planned() -> RunState {
    RunState::start("plan a trip", "claude", "m").with_plan(&["flights".to_string(), "hotel".to_string()])
}

fn statuses(state: &RunState) -> Vec<StepStatus> {
    state.steps.iter().map(|step| step.status).collect()
}

#[test]
fn a_new_run_starts_in_planning_with_no_steps() {
    let state = RunState::start("g", "claude", "m");
    assert_eq!(state.phase, Phase::Planning);
    assert!(state.steps.is_empty());
}

#[test]
fn a_plan_turns_every_task_into_a_pending_step_and_moves_to_working() {
    let state = planned();
    assert_eq!(state.phase, Phase::Working);
    assert_eq!(statuses(&state), vec![StepStatus::Pending, StepStatus::Pending]);
}

#[test]
fn only_the_targeted_step_changes() {
    let running = planned().step_running(1);
    assert_eq!(statuses(&running), vec![StepStatus::Pending, StepStatus::Running]);
}

#[test]
fn completed_work_feeds_only_finished_steps_to_the_next_agent() {
    let state = planned().step_done(0, "LIS to OPO").step_running(1);
    assert_eq!(state.completed_work(), "Task 1: flights\nResult: LIS to OPO");
}

#[test]
fn the_run_ends_as_done_with_a_summary_or_failed_with_the_error() {
    let done = planned().summarizing().finished("all set");
    assert_eq!((done.phase, done.summary.as_str()), (Phase::Done, "all set"));
    let failed = planned().failed("boom");
    assert_eq!((failed.phase, failed.error.as_str()), (Phase::Failed, "boom"));
}

#[test]
fn state_json_matches_the_shape_the_react_ui_reads() {
    let json = serde_json::to_value(planned().step_running(0)).unwrap();
    assert_eq!(json["phase"], "working");
    assert_eq!(json["steps"][0]["status"], "running");
    assert_eq!(json["steps"][1]["status"], "pending");
    for key in ["goal", "provider", "model", "summary", "error"] {
        assert!(json[key].is_string(), "missing {key}");
    }
}
