import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BIN = os.environ.get("AGENTINFRA_BIN", str(ROOT / "infra" / "bin" / "agentinfra"))


class InfraError(Exception):
    pass


def call(client: str, *args: str) -> Any:
    completed = subprocess.run([BIN, "--client", client, *args], capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise InfraError(completed.stderr.strip() or f"agentinfra exited {completed.returncode}")
    return json.loads(completed.stdout)


def log(client: str, level: str, message: str, **fields: object) -> None:
    call(client, "log", "write", level, message, *[f"--field={k}={json.dumps(v)}" for k, v in fields.items()])


def get_or_none(client: str, key: str) -> Any:
    try:
        return call(client, "state", "get", key)["value"]
    except InfraError as error:
        if "not_found" in str(error):
            return None
        raise
