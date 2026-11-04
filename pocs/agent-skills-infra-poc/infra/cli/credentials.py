import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cli.api import request_token

REFRESH_MARGIN_SECONDS = 30


@dataclass(frozen=True)
class Credentials:
    client_id: str
    client_secret: str


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


def load_credentials(secrets_dir: Path, client: str) -> Credentials:
    secret = os.environ.get("AGENTINFRA_CLIENT_SECRET")
    if secret:
        return Credentials(client, secret)
    path = secrets_dir / "credentials" / f"{client}.env"
    if not path.exists():
        raise FileNotFoundError(f"no credentials for client {client}, run agentinfra bootstrap")
    values = read_env_file(path)
    return Credentials(values["CLIENT_ID"], values["CLIENT_SECRET"])


def _cache_path(secrets_dir: Path, client: str) -> Path:
    return secrets_dir / "tokens" / f"{client}.json"


def cached_token(secrets_dir: Path, client: str, now: float) -> str | None:
    path = _cache_path(secrets_dir, client)
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except ValueError:
        return None
    if float(data.get("expires_at", 0)) - REFRESH_MARGIN_SECONDS <= now:
        return None
    return str(data["access_token"])


def fetch_token(secrets_dir: Path, auth_url: str, client: str) -> str:
    credentials = load_credentials(secrets_dir, client)
    response = request_token(auth_url, credentials.client_id, credentials.client_secret)
    path = _cache_path(secrets_dir, client)
    path.parent.mkdir(parents=True, exist_ok=True)
    expires_at = time.time() + int(response["expires_in"])
    path.write_text(json.dumps({"access_token": response["access_token"], "expires_at": expires_at}))
    path.chmod(0o600)
    return str(response["access_token"])


def get_token(secrets_dir: Path, auth_url: str, client: str, force: bool = False) -> str:
    token = None if force else cached_token(secrets_dir, client, time.time())
    return token or fetch_token(secrets_dir, auth_url, client)
