import base64
import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode

from common.settings import load_settings

SETTINGS = load_settings()


def secret_of(client: str) -> str:
    path = SETTINGS.secrets_dir / "credentials" / f"{client}.env"
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)["CLIENT_SECRET"]


def http(method: str, url: str, body: Any = None, headers: dict[str, str] | None = None, form: dict[str, str] | None = None) -> tuple[int, Any]:
    data = None
    all_headers = dict(headers or {})
    if form is not None:
        data = urlencode(form).encode()
        all_headers["content-type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode()
        all_headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=all_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        with error:
            raw = error.read()
        return error.code, json.loads(raw) if raw else None


def basic(client: str, secret: str | None = None) -> dict[str, str]:
    value = base64.b64encode(f"{client}:{secret if secret is not None else secret_of(client)}".encode()).decode()
    return {"authorization": f"Basic {value}"}


def token(client: str, scope: str = "") -> str:
    form = {"grant_type": "client_credentials"}
    if scope:
        form["scope"] = scope
    status, body = http("POST", f"{SETTINGS.auth_url}/oauth/token", headers=basic(client), form=form)
    assert status == 200, body
    return body["access_token"]


def bearer(client: str, scope: str = "") -> dict[str, str]:
    return {"authorization": f"Bearer {token(client, scope)}"}
