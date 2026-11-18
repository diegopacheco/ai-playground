import asyncio
import json

from agent_api.http import Api, parse_goal, to_json
from agent_api.run_state import start_run
from agent_api.runs import RunRow, RunView


class FakeStore:
    def __init__(self) -> None:
        self.started: list[str] = []

    def start(self, goal: str) -> str:
        self.started.append(goal)
        return "run_1"

    def runs(self) -> list[RunRow]:
        return [RunRow("run_1", "EXECUTING", "2026-11-17", "http://dash/run_1")]

    def run(self, run_id: str) -> RunView:
        return RunView(run_id, "EXECUTING", f"http://dash/{run_id}", start_run("g", "claude", "m"))


def call(api: Api, method: str, path: str, body: bytes = b"") -> tuple[int, object]:
    response = asyncio.run(api.handle(method, path, body))
    return response.status, json.loads(to_json(response.body))


def test_starting_a_run_returns_the_trigger_run_id_for_the_trimmed_goal() -> None:
    store = FakeStore()
    assert call(Api(store), "POST", "/runs", b'{"goal": "  trip "}') == (202, {"runId": "run_1"})
    assert store.started == ["trip"]


def test_a_run_without_a_goal_is_never_triggered() -> None:
    store = FakeStore()
    for body in (b"", b"{}", b'{"goal": "  "}', b"nope", b'{"goal": 3}'):
        assert call(Api(store), "POST", "/runs", body)[0] == 400
    assert store.started == []
    assert parse_goal(b'["goal"]') == ""


def test_a_run_view_carries_the_dashboard_link_and_the_live_state() -> None:
    status, body = call(Api(FakeStore()), "GET", "/runs/run_9")
    assert status == 200 and isinstance(body, dict)
    assert body["dashboard_url"] == "http://dash/run_9"
    assert body["state"]["phase"] == "planning"


def test_unknown_routes_are_404() -> None:
    assert call(Api(FakeStore()), "DELETE", "/runs")[0] == 404
    assert call(Api(FakeStore()), "GET", "/health") == (200, {"ok": True})
