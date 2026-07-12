from datetime import timedelta

from temporalio import workflow


@workflow.defn
class TrainModelWorkflow:
    @workflow.run
    async def run(self) -> dict:
        result = {}
        for batch in range(20):
            result = await workflow.execute_activity(
                "train_model_batch",
                batch,
                start_to_close_timeout=timedelta(minutes=1),
            )
        return result
