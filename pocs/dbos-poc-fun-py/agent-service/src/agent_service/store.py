from dbos import DBOS, SetWorkflowID

from agent_service.run_state import RunState
from agent_service.runs import RunRow
from agent_service.workflow import STATE_EVENT, WORKFLOW_NAME, agent_workflow


class DbosRunStore:
    def start(self, run_id: str, goal: str) -> None:
        with SetWorkflowID(run_id):
            DBOS.start_workflow(agent_workflow, goal)

    def runs(self) -> list[RunRow]:
        found = DBOS.list_workflows(name=WORKFLOW_NAME, sort_desc=True, limit=20, load_input=False, load_output=False)
        return [RunRow(item.workflow_id, item.status, item.created_at or 0) for item in found]

    def state(self, run_id: str) -> RunState | None:
        value = DBOS.get_event(run_id, STATE_EVENT, timeout_seconds=0)
        return value if isinstance(value, RunState) else None
