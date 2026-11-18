import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerConfig:
    api_url: str
    secret_key: str
    org_slug: str
    project_slug: str

    def dashboard_url(self, run_id: str) -> str:
        return f"{self.api_url}/orgs/{self.org_slug}/projects/{self.project_slug}/env/dev/runs/{run_id}"


def read_trigger_config(env: Mapping[str, str] = os.environ) -> TriggerConfig:
    names = ("TRIGGER_API_URL", "TRIGGER_SECRET_KEY", "TRIGGER_ORG_SLUG", "TRIGGER_PROJECT_SLUG")
    missing = [name for name in names if not env.get(name, "").strip()]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}, run ./scripts/start-all.sh to bootstrap trigger.dev")
    if not env["TRIGGER_SECRET_KEY"].startswith("tr_dev_"):
        raise ValueError("TRIGGER_SECRET_KEY must be the dev environment key, tr_dev_...")
    return TriggerConfig(env["TRIGGER_API_URL"].rstrip("/"), env["TRIGGER_SECRET_KEY"], env["TRIGGER_ORG_SLUG"], env["TRIGGER_PROJECT_SLUG"])
