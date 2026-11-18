from dataclasses import dataclass, replace
from typing import Literal

type Phase = Literal["planning", "working", "summarizing", "done", "failed"]
type StepStatus = Literal["pending", "running", "done"]


@dataclass(frozen=True)
class Step:
    task: str
    status: StepStatus
    output: str


@dataclass(frozen=True)
class RunState:
    goal: str
    provider: str
    model: str
    phase: Phase
    steps: tuple[Step, ...]
    summary: str
    error: str


def start_run(goal: str, provider: str, model: str) -> RunState:
    return RunState(goal, provider, model, "planning", (), "", "")


def with_plan(state: RunState, tasks: list[str]) -> RunState:
    return replace(state, phase="working", steps=tuple(Step(task, "pending", "") for task in tasks))


def step_running(state: RunState, index: int) -> RunState:
    steps = tuple(replace(step, status="running") if i == index else step for i, step in enumerate(state.steps))
    return replace(state, steps=steps)


def step_done(state: RunState, index: int, output: str) -> RunState:
    steps = tuple(replace(step, status="done", output=output) if i == index else step for i, step in enumerate(state.steps))
    return replace(state, steps=steps)


def summarizing(state: RunState) -> RunState:
    return replace(state, phase="summarizing")


def finished(state: RunState, summary: str) -> RunState:
    return replace(state, phase="done", summary=summary)


def failed(state: RunState, error: str) -> RunState:
    return replace(state, phase="failed", error=error)


def completed_work(state: RunState) -> str:
    done = [step for step in state.steps if step.status == "done"]
    return "\n\n".join(f"Task {i + 1}: {step.task}\nResult: {step.output}" for i, step in enumerate(done))
