import os
from collections.abc import Mapping
from dataclasses import dataclass

from agent_service.agents import Provider, is_provider


@dataclass(frozen=True)
class AgentConfig:
    provider: Provider
    model: str


def read_config(env: Mapping[str, str] = os.environ) -> AgentConfig:
    provider = env.get("AGENT_PROVIDER", "")
    model = env.get("AGENT_MODEL", "")
    if not is_provider(provider):
        raise ValueError(f'AGENT_PROVIDER must be claude, codex, agy or ollama, got "{provider}"')
    if not model.strip():
        raise ValueError("AGENT_MODEL is required")
    return AgentConfig(provider, model)


def database_url(env: Mapping[str, str] = os.environ) -> str:
    url = env.get("DBOS_SYSTEM_DATABASE_URL", "")
    if not url.startswith("postgresql://"):
        raise ValueError("DBOS_SYSTEM_DATABASE_URL must be a postgresql:// url")
    return url
