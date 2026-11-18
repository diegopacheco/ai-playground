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


def temporal_address(env: Mapping[str, str] = os.environ) -> str:
    address = env.get("TEMPORAL_ADDRESS", "")
    if ":" not in address:
        raise ValueError("TEMPORAL_ADDRESS must be host:port")
    return address


def temporal_ui_url(env: Mapping[str, str] = os.environ) -> str:
    url = env.get("TEMPORAL_UI_URL", "")
    if not url.startswith("http"):
        raise ValueError("TEMPORAL_UI_URL must be an http url")
    return url.rstrip("/")
