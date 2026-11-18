import pytest

from agent_service.plan import MAX_TASKS, parse_plan


def test_the_task_list_is_found_inside_chatty_model_output() -> None:
    assert parse_plan('Sure! Here it is:\n["a", " b "]\nGood luck') == ["a", "b"]


def test_the_plan_is_capped_so_a_run_cannot_explode_in_cost() -> None:
    assert len(parse_plan('["1","2","3","4","5","6"]')) == MAX_TASKS


def test_a_reply_without_tasks_fails_the_step_so_dbos_retries_it() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        parse_plan("I cannot help")
    with pytest.raises(ValueError, match="no tasks"):
        parse_plan('["", 3]')
