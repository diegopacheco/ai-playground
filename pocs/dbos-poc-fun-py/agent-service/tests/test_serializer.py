import json

import pytest

from agent_service.config import AgentConfig
from agent_service.run_state import RunState, step_running, with_plan
from agent_service.serializer import AgentJsonSerializer, RecordedError

serializer = AgentJsonSerializer()


def round_trip(value: object) -> object:
    return serializer.deserialize(serializer.serialize(value))


def test_workflow_state_survives_a_recovery_with_its_frozen_types() -> None:
    state = step_running(with_plan(RunState("goal", "claude", "m", "planning", (), "", ""), ["a", "b"]), 0)
    restored = round_trip(state)
    assert restored == state
    assert isinstance(restored, RunState) and isinstance(restored.steps, tuple)


def test_values_are_stored_as_plain_json_that_the_dashboard_can_read() -> None:
    stored = json.loads(serializer.serialize(AgentConfig("claude", "claude-sonnet-5")))
    assert stored == {"__type__": "agent_service.config.AgentConfig", "provider": "claude", "model": "claude-sonnet-5"}
    assert json.loads(serializer.serialize({"args": ("goal",), "kwargs": {}})) == {"args": ["goal"], "kwargs": {}}


def test_a_step_error_keeps_its_message_so_the_run_can_report_it() -> None:
    restored = round_trip(ValueError("model said no"))
    assert isinstance(restored, RecordedError)
    assert str(restored) == "model said no" and restored.kind == "builtins.ValueError"


def test_stored_data_can_never_load_classes_outside_the_service() -> None:
    with pytest.raises(TypeError):
        serializer.deserialize('{"__type__": "os.system"}')
    with pytest.raises(TypeError):
        serializer.serialize(object())
