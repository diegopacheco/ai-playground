import asyncio
from dataclasses import dataclass

from mcp_events.catalog import ALERT_RAISED, DEPLOY_STATUS_CHANGED, ENVIRONMENTS
from mcp_events.errors import invalid_params
from mcp_events.event_log import EventLog
from mcp_events.json_types import JsonObject

SEVERITY_BY_ENV = {"prod": "P1", "staging": "P2", "dev": "P3"}
PROGRESS = ("building", "testing")


@dataclass(frozen=True)
class Deploy:
    deploy_id: str
    service: str
    env: str
    simulate_failure: bool
    triggered_by: str

    def status_event(self, status: str) -> JsonObject:
        return {
            "deployId": self.deploy_id,
            "service": self.service,
            "env": self.env,
            "status": status,
            "triggeredBy": self.triggered_by,
        }

    def alert_event(self) -> JsonObject:
        return {
            "alertId": f"alr_{self.deploy_id.removeprefix('dep_')}",
            "severity": SEVERITY_BY_ENV[self.env],
            "message": f"deploy of {self.service} to {self.env} failed in testing",
            "deployId": self.deploy_id,
            "triggeredBy": self.triggered_by,
        }


class Deployer:
    def __init__(self, log: EventLog, step_seconds: float = 1.2) -> None:
        self._log = log
        self._step_seconds = step_seconds
        self._count = 0
        self._tasks: set[asyncio.Task[None]] = set()

    def start(self, service: str, env: str, simulate_failure: bool, triggered_by: str) -> Deploy:
        if env not in ENVIRONMENTS:
            raise invalid_params(f"env must be one of {', '.join(ENVIRONMENTS)}")
        self._count += 1
        deploy = Deploy(f"dep_{self._count:04d}", service, env, simulate_failure, triggered_by)
        self._log.append(DEPLOY_STATUS_CHANGED, deploy.status_event("queued"))
        task = asyncio.get_running_loop().create_task(self._run(deploy))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return deploy

    async def _run(self, deploy: Deploy) -> None:
        for status in PROGRESS:
            await asyncio.sleep(self._step_seconds)
            self._log.append(DEPLOY_STATUS_CHANGED, deploy.status_event(status))
        await asyncio.sleep(self._step_seconds)
        if deploy.simulate_failure:
            self._log.append(DEPLOY_STATUS_CHANGED, deploy.status_event("failed"))
            self._log.append(ALERT_RAISED, deploy.alert_event())
            return
        self._log.append(DEPLOY_STATUS_CHANGED, deploy.status_event("deployed"))

    async def drain(self) -> None:
        await asyncio.gather(*self._tasks)
