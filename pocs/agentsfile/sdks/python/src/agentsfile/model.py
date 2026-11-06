from dataclasses import dataclass, field
from enum import StrEnum


class Transport(StrEnum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass(frozen=True)
class Skill:
    name: str
    source: str
    description: str | None = None


@dataclass(frozen=True)
class Agent:
    name: str
    source: str
    description: str | None = None
    model: str | None = None
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class Command:
    name: str
    source: str
    description: str | None = None


@dataclass(frozen=True)
class Mcp:
    name: str
    transport: Transport
    description: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    name: str
    description: str | None = None
    skills: tuple[Skill, ...] = ()
    agents: tuple[Agent, ...] = ()
    commands: tuple[Command, ...] = ()
    mcps: tuple[Mcp, ...] = ()


@dataclass(frozen=True)
class Agentsfile:
    version: str
    profiles: tuple[Profile, ...] = ()
