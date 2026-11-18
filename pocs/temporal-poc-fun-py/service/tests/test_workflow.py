import asyncio
import shutil
import uuid
from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from agent_service.activities import PlanInput, SummaryInput, TaskInput
from agent_service.config import AgentConfig
from agent_service.run_state import RunState
from agent_service.workflow import STATE_MEMO, AgentWorkflow

TEMPORAL = shutil.which("temporal")
pytestmark = pytest.mark.skipif(TEMPORAL is None, reason="the temporal CLI is needed for a local dev server")


def fake_activities(calls: list[str], fail_task: bool = False) -> list:
    @activity.defn(name="read_config")
    async def read_config() -> AgentConfig:
        return AgentConfig("claude", "m")

    @activity.defn(name="plan_goal")
    async def plan_goal(request: PlanInput) -> list[str]:
        calls.append("plan")
        return ["a", "b"]

    @activity.defn(name="execute_task")
    async def execute_task(request: TaskInput) -> str:
        calls.append(f"task {request.task} after {request.previous_work}")
        if fail_task:
            raise RuntimeError("agent cli failed")
        return f"out-{request.task}"

    @activity.defn(name="summarize_run")
    async def summarize_run(request: SummaryInput) -> str:
        calls.append("summary")
        return f"final from {request.work.count('Task')} tasks"

    return [read_config, plan_goal, execute_task, summarize_run]


async def run_workflow(calls: list[str], fail_task: bool) -> tuple[str | None, RunState | None]:
    async with await WorkflowEnvironment.start_local(dev_server_existing_path=TEMPORAL, ui=False) as env:
        queue = f"q-{uuid.uuid4()}"
        async with Worker(env.client, task_queue=queue, workflows=[AgentWorkflow], activities=fake_activities(calls, fail_task)):
            handle = await env.client.start_workflow(AgentWorkflow.run, "goal", id=f"wf-{uuid.uuid4()}", task_queue=queue, execution_timeout=timedelta(seconds=60))
            try:
                result: str | None = await handle.result()
            except WorkflowFailureError:
                result = None
            state = await (await handle.describe()).memo_value(STATE_MEMO, None, type_hint=RunState)
            return result, state


def test_the_workflow_plans_runs_each_task_with_earlier_work_and_summarizes() -> None:
    calls: list[str] = []
    result, state = asyncio.run(run_workflow(calls, fail_task=False))
    assert calls == ["plan", "task a after none yet", "task b after Task 1: a\nResult: out-a", "summary"]
    assert result == "final from 2 tasks"
    assert state is not None and state.phase == "done" and [step.output for step in state.steps] == ["out-a", "out-b"]


def test_a_step_that_keeps_failing_is_retried_3_times_then_the_run_is_failed_with_the_error() -> None:
    calls: list[str] = []
    result, state = asyncio.run(run_workflow(calls, fail_task=True))
    assert result is None
    assert calls.count("task a after none yet") == 3
    assert state is not None and state.phase == "failed" and "agent cli failed" in state.error
