import json
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlencode

type Transport = Callable[[str, str, dict[str, str], bytes | None], bytes]

WORKFLOW_TASK = "agent-workflow"


class TriggerError(Exception):
    pass


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise TriggerError(f"{method} {url} returned {error.code}: {error.read().decode(errors='replace')[:300]}") from error


class TriggerClient:
    def __init__(self, api_url: str, secret_key: str, transport: Transport = urllib_transport) -> None:
        self.api_url = api_url
        self.secret_key = secret_key
        self.transport = transport

    def request(self, method: str, path: str, body: object = None) -> object:
        headers = {"authorization": f"Bearer {self.secret_key}", "content-type": "application/json"}
        data = None if body is None else json.dumps(body).encode()
        return json.loads(self.transport(method, f"{self.api_url}{path}", headers, data))

    def trigger(self, task: str, payload: object) -> str:
        result = self.request("POST", f"/api/v1/tasks/{task}/trigger", {"payload": payload})
        run_id = result.get("id") if isinstance(result, dict) else None
        if not isinstance(run_id, str):
            raise TriggerError(f"trigger.dev returned no run id: {result}")
        return run_id

    def retrieve(self, run_id: str) -> object:
        return self.request("GET", f"/api/v3/runs/{run_id}")

    def list_runs(self, task: str, limit: int = 20) -> object:
        query = urlencode({"filter[taskIdentifier]": task, "page[size]": limit})
        return self.request("GET", f"/api/v1/runs?{query}")
