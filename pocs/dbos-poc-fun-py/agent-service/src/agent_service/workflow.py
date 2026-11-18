from dbos import DBOS

from agent_service.agents import agent_for
from agent_service.config import AgentConfig, read_config
from agent_service.plan import parse_plan
from agent_service.prompts import load_prompt
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
STATE_EVENT = "state"


def ask(config: AgentConfig, prompt: str) -> str:
    return agent_for(config.provider).call(config.model, prompt).strip()


@DBOS.step(name="read config")
def read_agent_config() -> AgentConfig:
    return read_config()


@DBOS.step(name="plan", retries_allowed=True, max_attempts=3, interval_seconds=2)
def plan_goal(config: AgentConfig, goal: str) -> list[str]:
    return parse_plan(ask(config, load_prompt("plan-goal", {"goal": goal})))


@DBOS.step(name="task", retries_allowed=True, max_attempts=3, interval_seconds=2)
def execute_task(config: AgentConfig, goal: str, task: str, previous_work: str) -> str:
    return ask(config, load_prompt("execute-task", {"goal": goal, "task": task, "previous_work": previous_work}))


@DBOS.step(name="summarize", retries_allowed=True, max_attempts=3, interval_seconds=2)
def summarize_run(config: AgentConfig, goal: str, work: str) -> str:
    return ask(config, load_prompt("summarize-run", {"goal": goal, "work": work}))


def publish(state: RunState) -> RunState:
    DBOS.set_event(STATE_EVENT, state)
    return state


@DBOS.workflow(name=WORKFLOW_NAME)
def agent_workflow(goal: str) -> str:
    config = read_agent_config()
    state = publish(start_run(goal, config.provider, config.model))
    try:
        tasks = plan_goal(config, goal)
        state = publish(with_plan(state, tasks))
        for index, task in enumerate(tasks):
            state = publish(step_running(state, index))
            output = execute_task(config, goal, task, completed_work(state) or "none yet")
            state = publish(step_done(state, index, output))
        state = publish(summarizing(state))
        summary = summarize_run(config, goal, completed_work(state))
        publish(finished(state, summary))
        return summary
    except Exception as error:
        publish(failed(state, str(error)))
        raise
