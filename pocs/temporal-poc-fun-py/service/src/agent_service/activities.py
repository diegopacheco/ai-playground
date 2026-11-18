import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from temporalio import activity

from agent_service.agents import agent_for
from agent_service.config import AgentConfig, read_config
from agent_service.plan import parse_plan
from agent_service.prompts import load_prompt

HEARTBEAT_SECONDS = 2.0


@dataclass(frozen=True)
class PlanInput:
    config: AgentConfig
    goal: str


@dataclass(frozen=True)
class TaskInput:
    config: AgentConfig
    goal: str
    task: str
    previous_work: str


@dataclass(frozen=True)
class SummaryInput:
    config: AgentConfig
    goal: str
    work: str


def ask(config: AgentConfig, prompt: str) -> str:
    return agent_for(config.provider).call(config.model, prompt).strip()


async def with_heartbeat[T](work: Callable[[], T], beat: Callable[[], None] = activity.heartbeat) -> T:
    running = asyncio.ensure_future(asyncio.to_thread(work))
    while True:
        done, _ = await asyncio.wait({running}, timeout=HEARTBEAT_SECONDS)
        if done:
            return running.result()
        beat()


@activity.defn(name="read_config")
async def read_agent_config() -> AgentConfig:
    return read_config()


@activity.defn(name="plan_goal")
async def plan_goal(request: PlanInput) -> list[str]:
    return await with_heartbeat(lambda: parse_plan(ask(request.config, load_prompt("plan-goal", {"goal": request.goal}))))


@activity.defn(name="execute_task")
async def execute_task(request: TaskInput) -> str:
    params = {"goal": request.goal, "task": request.task, "previous_work": request.previous_work}
    return await with_heartbeat(lambda: ask(request.config, load_prompt("execute-task", params)))


@activity.defn(name="summarize_run")
async def summarize_run(request: SummaryInput) -> str:
    params = {"goal": request.goal, "work": request.work}
    return await with_heartbeat(lambda: ask(request.config, load_prompt("summarize-run", params)))


ACTIVITIES = [read_agent_config, plan_goal, execute_task, summarize_run]
