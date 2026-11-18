import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from agent_service.activities import ACTIVITIES
from agent_service.config import read_config, temporal_address
from agent_service.workflow import TASK_QUEUE, AgentWorkflow


async def main() -> None:
    config = read_config()
    client = await Client.connect(temporal_address(), identity="agent-worker")
    worker = Worker(client, task_queue=TASK_QUEUE, workflows=[AgentWorkflow], activities=ACTIVITIES, identity="agent-worker")
    print(f"worker polling {TASK_QUEUE} using {config.provider} {config.model}", flush=True)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
