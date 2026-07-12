import asyncio
import os
from datetime import datetime, timezone

from temporalio.client import Client

from model_pipeline.workflow import TrainModelWorkflow


async def main() -> None:
    client = await Client.connect(os.getenv("TEMPORAL_ADDRESS", "temporal:7233"))
    workflow_id = f"iris-training-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Starting workflow {workflow_id}")
    result = await client.execute_workflow(TrainModelWorkflow.run, id=workflow_id, task_queue="model-training")
    print(f"Training completed: {result}")


if __name__ == "__main__":
    asyncio.run(main())
