from agent_service.run_state import (
    completed_work,
    failed,
    finished,
    start_run,
    step_done,
    step_running,
    summarizing,
    with_plan,
)


def test_a_run_moves_from_planning_to_done_without_mutating_earlier_states() -> None:
    start = start_run("g", "claude", "m")
    planned = with_plan(start, ["a", "b"])
    running = step_running(planned, 0)
    done = step_done(running, 0, "out-a")
    assert start.phase == "planning" and start.steps == ()
    assert [step.status for step in planned.steps] == ["pending", "pending"]
    assert [step.status for step in running.steps] == ["running", "pending"]
    assert done.steps[0].output == "out-a" and planned.steps[0].output == ""
    assert summarizing(done).phase == "summarizing"
    assert finished(done, "final").summary == "final"


def test_each_worker_only_sees_finished_work() -> None:
    state = step_running(step_done(with_plan(start_run("g", "c", "m"), ["a", "b"]), 0, "out-a"), 1)
    assert completed_work(state) == "Task 1: a\nResult: out-a"


def test_a_failure_keeps_the_progress_made_so_far() -> None:
    state = failed(step_done(with_plan(start_run("g", "c", "m"), ["a"]), 0, "x"), "boom")
    assert state.phase == "failed" and state.error == "boom" and state.steps[0].output == "x"
