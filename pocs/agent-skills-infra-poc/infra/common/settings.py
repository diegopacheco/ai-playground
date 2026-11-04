import os
from dataclasses import dataclass
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = INFRA_DIR.parent
PORTS_FILE = ROOT_DIR / "scripts" / "ports.env"


def read_ports(path: Path = PORTS_FILE) -> dict[str, int]:
    ports: dict[str, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and "=" in line:
            name, value = line.split("=", 1)
            ports[name.strip()] = int(value.strip())
    return ports


@dataclass(frozen=True)
class Settings:
    secrets_dir: Path
    issuer: str
    auth_url: str
    state_url: str
    logs_url: str
    evals_url: str
    loki_url: str
    database_url: str
    token_ttl_seconds: int


def load_settings() -> Settings:
    ports = read_ports()
    host = "http://127.0.0.1"
    auth_url = os.environ.get("AGENTINFRA_AUTH_URL", f"{host}:{ports['auth']}")
    return Settings(
        secrets_dir=Path(os.environ.get("AGENTINFRA_SECRETS_DIR", str(INFRA_DIR / ".secrets"))),
        issuer=auth_url,
        auth_url=auth_url,
        state_url=os.environ.get("AGENTINFRA_STATE_URL", f"{host}:{ports['state']}"),
        logs_url=os.environ.get("AGENTINFRA_LOGS_URL", f"{host}:{ports['logs']}"),
        evals_url=os.environ.get("AGENTINFRA_EVALS_URL", f"{host}:{ports['evals']}"),
        loki_url=os.environ.get("AGENTINFRA_LOKI_URL", f"{host}:{ports['loki']}"),
        database_url=os.environ.get(
            "AGENTINFRA_DATABASE_URL",
            f"postgresql://agentinfra_app:agentinfra_app@127.0.0.1:{ports['postgres']}/agentinfra",
        ),
        token_ttl_seconds=int(os.environ.get("AGENTINFRA_TOKEN_TTL", "900")),
    )
