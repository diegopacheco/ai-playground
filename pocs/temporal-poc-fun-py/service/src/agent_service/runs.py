from dataclasses import dataclass
from typing import Protocol

from agent_service.run_state import RunState


@dataclass(frozen=True)
class RunRow:
    run_id: str
    status: str
    created_at: str
    temporal_url: str


@dataclass(frozen=True)
class RunView:
    run_id: str
    status: str
    temporal_url: str
    state: RunState | None


class RunStore(Protocol):
    async def start(self, run_id: str, goal: str) -> None: ...
    async def runs(self) -> list[RunRow]: ...
    async def run(self, run_id: str) -> RunView | None: ...


def workflow_url(ui_url: str, run_id: str) -> str:
    return f"{ui_url}/namespaces/default/workflows/{run_id}"
