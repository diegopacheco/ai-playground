import asyncio
import json

from agent_service.http import Api, base36, parse_goal, to_json
from agent_service.run_state import start_run
from agent_service.runs import RunRow, RunView, workflow_url


class FakeStore:
    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []

    async def start(self, run_id: str, goal: str) -> None:
        self.started.append((run_id, goal))

    async def runs(self) -> list[RunRow]:
        return [RunRow("run-1", "RUNNING", "2026-11-17T10:00:00", "http://ui/run-1")]

    async def run(self, run_id: str) -> RunView | None:
        return RunView(run_id, "RUNNING", "http://ui/run-1", start_run("g", "claude", "m")) if run_id == "run-1" else None


def call(store: FakeStore, method: str, path: str, body: bytes = b"") -> tuple[int, object]:
    response = asyncio.run(Api(lambda: store, lambda: "run-x").handle(method, path, body))
    return response.status, json.loads(to_json(response.body))


def test_starting_a_run_uses_the_run_id_as_the_workflow_id_with_the_trimmed_goal() -> None:
    store = FakeStore()
    assert call(store, "POST", "/runs", b'{"goal": "  trip  "}') == (202, {"runId": "run-x"})
    assert store.started == [("run-x", "trip")]


def test_a_run_without_a_goal_is_never_started() -> None:
    store = FakeStore()
    for body in (b"", b"{}", b'{"goal": " "}', b"not json", b'{"goal": 7}'):
        assert call(store, "POST", "/runs", body)[0] == 400
    assert store.started == []
    assert parse_goal(b'["goal"]') == ""


def test_a_run_view_has_the_live_state_and_unknown_runs_are_404() -> None:
    status, body = call(FakeStore(), "GET", "/runs/run-1")
    assert status == 200 and isinstance(body, dict)
    assert body["state"]["phase"] == "planning" and body["temporal_url"] == "http://ui/run-1"
    assert call(FakeStore(), "GET", "/runs/nope")[0] == 404
    assert call(FakeStore(), "DELETE", "/runs")[0] == 404


def test_the_temporal_button_deep_links_to_the_workflow_page() -> None:
    assert workflow_url("http://localhost:8233", "run-1") == "http://localhost:8233/namespaces/default/workflows/run-1"


def test_run_ids_follow_time_so_each_goal_is_a_new_workflow() -> None:
    assert base36(0) == "0" and base36(36) == "10"
