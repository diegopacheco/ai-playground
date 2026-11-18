import json

import pytest

from agent_api.config import TriggerConfig, read_trigger_config
from agent_api.runs import TriggerRunStore, parse_state
from agent_api.trigger_client import TriggerClient, TriggerError

CONFIG = TriggerConfig("http://localhost:8030", "tr_dev_abc", "agent-poc", "agent-workflow-x1")
STATE = {
    "goal": "g", "provider": "claude", "model": "m", "phase": "working",
    "steps": [{"task": "a", "status": "done", "output": "out"}, {"task": "b", "status": "running", "output": ""}],
    "summary": "", "error": "",
}


class FakeTransport:
    def __init__(self, replies: dict[str, object]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
        self.calls.append((method, url, headers, body))
        path = url.removeprefix(CONFIG.api_url)
        return json.dumps(next(reply for prefix, reply in self.replies.items() if path.startswith(prefix))).encode()


def store(replies: dict[str, object]) -> tuple[TriggerRunStore, FakeTransport]:
    transport = FakeTransport(replies)
    return TriggerRunStore(CONFIG, TriggerClient(CONFIG.api_url, CONFIG.secret_key, transport)), transport


def test_starting_a_run_triggers_the_workflow_task_with_the_dev_secret_key() -> None:
    runs, transport = store({"/api/v1/tasks/agent-workflow/trigger": {"id": "run_1"}})
    assert runs.start("trip") == "run_1"
    method, url, headers, body = transport.calls[0]
    assert method == "POST" and url.endswith("/api/v1/tasks/agent-workflow/trigger")
    assert headers["authorization"] == "Bearer tr_dev_abc"
    assert json.loads(body or b"") == {"payload": {"goal": "trip"}}


def test_a_trigger_reply_without_run_id_fails_loud() -> None:
    runs, _ = store({"/api/v1/tasks": {"error": "nope"}})
    with pytest.raises(TriggerError, match="no run id"):
        runs.start("trip")


def test_the_run_list_only_asks_for_workflow_runs_and_links_each_to_the_dashboard() -> None:
    runs, transport = store({"/api/v1/runs": {"data": [{"id": "run_1", "status": "EXECUTING", "createdAt": "2026-11-17T10:00:00Z"}]}})
    [row] = runs.runs()
    assert "filter%5BtaskIdentifier%5D=agent-workflow" in transport.calls[0][1]
    assert row.status == "EXECUTING"
    assert row.dashboard_url == "http://localhost:8030/orgs/agent-poc/projects/agent-workflow-x1/env/dev/runs/run_1"


def test_live_progress_comes_from_the_run_metadata_the_task_publishes() -> None:
    runs, _ = store({"/api/v3/runs/run_1": {"id": "run_1", "status": "EXECUTING", "metadata": {"state": STATE}}})
    view = runs.run("run_1")
    assert view.status == "EXECUTING" and view.state is not None
    assert [step.status for step in view.state.steps] == ["done", "running"]


def test_metadata_that_is_missing_or_malformed_means_no_state_yet() -> None:
    assert parse_state(None) is None
    assert parse_state({"state": {"phase": "dancing"}}) is None
    broken = parse_state({"state": {**STATE, "steps": [{"task": "x", "status": "weird"}, "junk"]}})
    assert broken is not None and broken.steps == ()


def test_the_api_refuses_to_start_without_a_bootstrapped_dev_key() -> None:
    with pytest.raises(ValueError, match="TRIGGER_ORG_SLUG"):
        read_trigger_config({"TRIGGER_API_URL": "u", "TRIGGER_SECRET_KEY": "tr_dev_x", "TRIGGER_PROJECT_SLUG": "p"})
    with pytest.raises(ValueError, match="tr_dev_"):
        read_trigger_config({"TRIGGER_API_URL": "u", "TRIGGER_SECRET_KEY": "tr_prod_x", "TRIGGER_ORG_SLUG": "o", "TRIGGER_PROJECT_SLUG": "p"})
