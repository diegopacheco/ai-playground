from dataclasses import dataclass
from typing import Protocol, cast, get_args

from agent_api.config import TriggerConfig
from agent_api.run_state import Phase, RunState, Step, StepStatus
from agent_api.trigger_client import WORKFLOW_TASK, TriggerClient

PHASES: tuple[str, ...] = get_args(Phase.__value__)
STEP_STATUSES: tuple[str, ...] = get_args(StepStatus.__value__)


@dataclass(frozen=True)
class RunRow:
    run_id: str
    status: str
    created_at: str
    dashboard_url: str


@dataclass(frozen=True)
class RunView:
    run_id: str
    status: str
    dashboard_url: str
    state: RunState | None


class RunStore(Protocol):
    def start(self, goal: str) -> str: ...
    def runs(self) -> list[RunRow]: ...
    def run(self, run_id: str) -> RunView: ...


def text(value: object) -> str:
    return value if isinstance(value, str) else ""


def parse_step(value: object) -> Step | None:
    if not isinstance(value, dict):
        return None
    status = text(value.get("status"))
    if status not in STEP_STATUSES:
        return None
    return Step(text(value.get("task")), cast(StepStatus, status), text(value.get("output")))


def parse_state(metadata: object) -> RunState | None:
    state = metadata.get("state") if isinstance(metadata, dict) else None
    if not isinstance(state, dict) or text(state.get("phase")) not in PHASES:
        return None
    raw_steps = state.get("steps")
    steps = tuple(step for step in map(parse_step, raw_steps if isinstance(raw_steps, list) else []) if step is not None)
    return RunState(
        text(state.get("goal")),
        text(state.get("provider")),
        text(state.get("model")),
        cast(Phase, text(state.get("phase"))),
        steps,
        text(state.get("summary")),
        text(state.get("error")),
    )


class TriggerRunStore:
    def __init__(self, config: TriggerConfig, client: TriggerClient) -> None:
        self.config = config
        self.client = client

    def start(self, goal: str) -> str:
        return self.client.trigger(WORKFLOW_TASK, {"goal": goal})

    def runs(self) -> list[RunRow]:
        result = self.client.list_runs(WORKFLOW_TASK)
        data = result.get("data") if isinstance(result, dict) else None
        rows = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
        return [
            RunRow(text(row.get("id")), text(row.get("status")), text(row.get("createdAt")), self.config.dashboard_url(text(row.get("id"))))
            for row in rows
        ]

    def run(self, run_id: str) -> RunView:
        result = self.client.retrieve(run_id)
        run = result if isinstance(result, dict) else {}
        return RunView(run_id, text(run.get("status")), self.config.dashboard_url(run_id), parse_state(run.get("metadata")))
