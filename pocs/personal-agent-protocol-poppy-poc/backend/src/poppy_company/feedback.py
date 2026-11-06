from dataclasses import dataclass

from poppy_company.errors import invalid_request
from poppy_company.json_types import JsonObject

MAX_TASK = 200
MAX_DESCRIPTION = 2000


@dataclass(frozen=True)
class Report:
    client_id: str
    task: str
    description: str


class FeedbackBox:
    def __init__(self) -> None:
        self.reports: list[Report] = []

    def submit(self, client_id: str, body: JsonObject) -> Report:
        task, description = body.get("task"), body.get("description")
        if not isinstance(task, str) or not task.strip() or len(task) > MAX_TASK:
            raise invalid_request(f"task must be 1 to {MAX_TASK} characters")
        if not isinstance(description, str) or not description.strip() or len(description) > MAX_DESCRIPTION:
            raise invalid_request(f"description must be 1 to {MAX_DESCRIPTION} characters")
        report = Report(client_id, task.strip(), description.strip())
        self.reports.append(report)
        return report
