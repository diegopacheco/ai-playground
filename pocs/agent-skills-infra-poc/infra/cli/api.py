import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode


class ApiError(Exception):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"{status} {body}")
        self.status = status
        self.body = body


def _send(request: urllib.request.Request) -> Any:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        with error:
            raw_error = error.read()
        try:
            body: Any = json.loads(raw_error)
        except ValueError:
            body = raw_error.decode(errors="replace")
        raise ApiError(error.code, body) from error
    except urllib.error.URLError as error:
        raise ApiError(0, {"error": "unreachable", "error_description": str(error.reason)}) from error
    return json.loads(raw) if raw else None


def request_token(auth_url: str, client_id: str, client_secret: str) -> dict[str, Any]:
    basic = base64.b64encode(f"{quote(client_id)}:{quote(client_secret)}".encode()).decode()
    request = urllib.request.Request(
        f"{auth_url}/oauth/token",
        data=urlencode({"grant_type": "client_credentials"}).encode(),
        headers={"authorization": f"Basic {basic}", "content-type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    return _send(request)


@dataclass(frozen=True)
class BearerClient:
    base_url: str
    token: str

    def call(self, method: str, path: str, body: Any = None, query: dict[str, str] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        clean = {k: v for k, v in (query or {}).items() if v is not None}
        if clean:
            url += "?" + urlencode(clean)
        headers = {"authorization": f"Bearer {self.token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        return _send(urllib.request.Request(url, data=data, headers=headers, method=method))
