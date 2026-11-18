from dbos import DBOS, SetWorkflowID

from agent_service.run_state import RunState
from agent_service.runs import Journal, JournalStep, RunRow, describe
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

    def journal(self, run_id: str) -> Journal | None:
        status = DBOS.get_workflow_status(run_id)
        if status is None:
            return None
        steps = tuple(
            JournalStep(
                step["function_id"],
                step["function_name"],
                describe(step["output"]),
                describe(step["error"]),
                step["started_at_epoch_ms"],
                step["completed_at_epoch_ms"],
            )
            for step in DBOS.list_workflow_steps(run_id)
        )
        return Journal(run_id, status.status, status.executor_id or "", status.app_version or "", steps)
