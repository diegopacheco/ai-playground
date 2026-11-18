import json
from dataclasses import dataclass
from typing import Protocol

from agent_service.run_state import RunState

MAX_OUTPUT = 4000


@dataclass(frozen=True)
class RunRow:
    run_id: str
    status: str
    created_at: int


@dataclass(frozen=True)
class JournalStep:
    id: int
    name: str
    output: str
    error: str
    started_at: int | None
    completed_at: int | None


@dataclass(frozen=True)
class Journal:
    run_id: str
    status: str
    executor: str
    app_version: str
    steps: tuple[JournalStep, ...]


class RunStore(Protocol):
    def start(self, run_id: str, goal: str) -> None: ...
    def runs(self) -> list[RunRow]: ...
    def state(self, run_id: str) -> RunState | None: ...
    def journal(self, run_id: str) -> Journal | None: ...


def describe(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:MAX_OUTPUT]
    if isinstance(value, list | dict):
        return json.dumps(value)[:MAX_OUTPUT]
    return repr(value)[:MAX_OUTPUT]
