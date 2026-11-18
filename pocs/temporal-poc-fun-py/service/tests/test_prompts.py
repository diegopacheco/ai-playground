import pytest

from agent_service.prompts import PROMPTS_DIR, load_prompt, render_prompt

TEMPLATE = "---\nname: t\nparams: $goal, $goal_size\n---\n\nDo $goal in $goal_size.\n"


def test_parameters_are_replaced_longest_first_so_prefixes_do_not_clash() -> None:
    assert render_prompt(TEMPLATE, {"goal": "x", "goal_size": "2 steps"}) == "Do x in 2 steps."


def test_an_empty_parameter_fails_loud_instead_of_sending_a_broken_prompt() -> None:
    with pytest.raises(ValueError, match=r"\$goal_size"):
        render_prompt(TEMPLATE, {"goal": "x", "goal_size": " "})


def test_a_prompt_without_header_is_rejected() -> None:
    with pytest.raises(ValueError, match="header"):
        render_prompt("Do $goal", {"goal": "x"})


def test_every_prompt_the_workflow_uses_loads_from_the_prompts_folder() -> None:
    assert (PROMPTS_DIR / "README.md").exists()
    assert "Lisbon" in load_prompt("plan-goal", {"goal": "Lisbon"})
    assert "task-a" in load_prompt("execute-task", {"goal": "g", "task": "task-a", "previous_work": "none yet"})
    assert "work-b" in load_prompt("summarize-run", {"goal": "g", "work": "work-b"})
