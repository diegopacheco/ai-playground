import asyncio
import json

from agent_service.http import Api, base36, parse_goal, to_json
from agent_service.run_state import RunState, start_run
from agent_service.runs import RunRow


class FakeStore:
    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []

    def start(self, run_id: str, goal: str) -> None:
        self.started.append((run_id, goal))

    def runs(self) -> list[RunRow]:
        return [RunRow("run-1", "PENDING", 1)]

    def state(self, run_id: str) -> RunState | None:
        return start_run("g", "claude", "m") if run_id == "run-1" else None


def call(api: Api, method: str, path: str, body: bytes = b"") -> tuple[int, object]:
    response = asyncio.run(api.handle(method, path, body))
    return response.status, json.loads(to_json(response.body))


def test_starting_a_run_uses_a_fresh_id_and_the_trimmed_goal() -> None:
    store = FakeStore()
    status, body = call(Api(store, lambda: "run-x"), "POST", "/runs", b'{"goal": "  trip  "}')
    assert status == 202 and body == {"runId": "run-x"}
    assert store.started == [("run-x", "trip")]


def test_a_run_without_a_goal_is_never_started() -> None:
    store = FakeStore()
    for body in (b"", b"{}", b'{"goal": " "}', b"not json", b'["goal"]'):
        status, _ = call(Api(store), "POST", "/runs", body)
        assert status == 400
    assert store.started == []


def test_state_is_null_until_the_workflow_publishes_it() -> None:
    api = Api(FakeStore())
    assert call(api, "GET", "/runs/run-1")[1] == {
        "goal": "g", "provider": "claude", "model": "m", "phase": "planning", "steps": [], "summary": "", "error": "",
    }
    assert call(api, "GET", "/runs/run-2") == (200, None)


def test_unknown_routes_are_404() -> None:
    assert call(Api(FakeStore()), "DELETE", "/runs")[0] == 404
    assert call(Api(FakeStore()), "GET", "/runs/run-1/journal")[0] == 404


def test_run_ids_sort_by_time_and_a_non_string_goal_is_rejected() -> None:
    assert base36(0) == "0" and base36(35) == "z" and base36(36) == "10"
    assert parse_goal(b'{"goal": 1}') == ""
