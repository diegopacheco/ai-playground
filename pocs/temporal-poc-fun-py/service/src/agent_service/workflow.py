from collections.abc import Awaitable, Callable
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from agent_service.activities import (
        PlanInput,
        SummaryInput,
        TaskInput,
        execute_task,
        plan_goal,
        read_agent_config,
        summarize_run,
    )
    from agent_service.run_state import (
        RunState,
        completed_work,
        failed,
        finished,
        start_run,
        step_done,
        step_running,
        summarizing,
        with_plan,
    )

WORKFLOW_NAME = "AgentWorkflow"
TASK_QUEUE = "agent-tasks"
STATE_MEMO = "state"
RETRY = RetryPolicy(initial_interval=timedelta(seconds=2), maximum_attempts=3)


async def agent_call[I, O](activity: Callable[[I], Awaitable[O]], request: I) -> O:
    return await workflow.execute_activity(
        activity,
        request,
        start_to_close_timeout=timedelta(minutes=5),
        heartbeat_timeout=timedelta(seconds=10),
        retry_policy=RETRY,
    )


@workflow.defn(name=WORKFLOW_NAME)
class AgentWorkflow:
    def __init__(self) -> None:
        self.state: RunState | None = None

    def publish(self, state: RunState) -> RunState:
        self.state = state
        workflow.upsert_memo({STATE_MEMO: state})
        return state

    @workflow.query
    def status(self) -> RunState | None:
        return self.state

    @workflow.run
    async def run(self, goal: str) -> str:
        config = await workflow.execute_activity(read_agent_config, start_to_close_timeout=timedelta(seconds=10))
        state = self.publish(start_run(goal, config.provider, config.model))
        try:
            tasks = await agent_call(plan_goal, PlanInput(config, goal))
            state = self.publish(with_plan(state, tasks))
            for index, task in enumerate(tasks):
                state = self.publish(step_running(state, index))
                request = TaskInput(config, goal, task, completed_work(state) or "none yet")
                output = await agent_call(execute_task, request)
                state = self.publish(step_done(state, index, output))
            state = self.publish(summarizing(state))
            summary = await agent_call(summarize_run, SummaryInput(config, goal, completed_work(state)))
            self.publish(finished(state, summary))
            return summary
        except Exception as error:
            self.publish(failed(state, str(error.__cause__ or error)))
            raise
