from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode

from agent_service.run_state import RunState
from agent_service.runs import RunRow, RunView, workflow_url
from agent_service.workflow import STATE_MEMO, TASK_QUEUE, WORKFLOW_NAME, AgentWorkflow


class TemporalRunStore:
    def __init__(self, client: Client, ui_url: str) -> None:
        self.client = client
        self.ui_url = ui_url

    async def start(self, run_id: str, goal: str) -> None:
        await self.client.start_workflow(AgentWorkflow.run, goal, id=run_id, task_queue=TASK_QUEUE)

    async def runs(self) -> list[RunRow]:
        found = self.client.list_workflows(f'WorkflowType = "{WORKFLOW_NAME}"', limit=20)
        return [
            RunRow(item.id, item.status.name if item.status else "UNKNOWN", item.start_time.isoformat(), workflow_url(self.ui_url, item.id))
            async for item in found
        ]

    async def run(self, run_id: str) -> RunView | None:
        try:
            description = await self.client.get_workflow_handle(run_id).describe()
        except RPCError as error:
            if error.status == RPCStatusCode.NOT_FOUND:
                return None
            raise
        state = await description.memo_value(STATE_MEMO, None, type_hint=RunState)
        status = description.status.name if description.status else "UNKNOWN"
        return RunView(run_id, status, workflow_url(self.ui_url, run_id), state if isinstance(state, RunState) else None)
