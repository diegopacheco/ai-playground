import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from model_pipeline.training import train_model_batch
from model_pipeline.workflow import TrainModelWorkflow


async def main() -> None:
    client = await Client.connect(os.getenv("TEMPORAL_ADDRESS", "localhost:7233"))
    worker = Worker(client, task_queue="model-training", workflows=[TrainModelWorkflow], activities=[train_model_batch])
    print("Training worker is ready")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
