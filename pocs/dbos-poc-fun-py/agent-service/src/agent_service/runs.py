from dataclasses import dataclass
from typing import Protocol

from agent_service.run_state import RunState


@dataclass(frozen=True)
class RunRow:
    run_id: str
    status: str
    created_at: int


class RunStore(Protocol):
    def start(self, run_id: str, goal: str) -> None: ...
    def runs(self) -> list[RunRow]: ...
    def state(self, run_id: str) -> RunState | None: ...
