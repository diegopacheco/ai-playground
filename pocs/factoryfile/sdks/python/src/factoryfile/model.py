from dataclasses import dataclass


@dataclass(frozen=True)
class Agent:
    name: str
    description: str | None = None
    role: str | None = None
    model: str | None = None
    tools: tuple[str, ...] = ()
    source: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class Stage:
    name: str
    description: str | None = None
    agents: tuple[str, ...] = ()
    next: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    content: str | None = None


@dataclass(frozen=True)
class Loop:
    name: str
    from_: str
    to: str
    description: str | None = None
    until: str | None = None
    max: int | None = None


@dataclass(frozen=True)
class Factory:
    name: str
    description: str | None = None
    entry: str | None = None
    agents: tuple[Agent, ...] = ()
    stages: tuple[Stage, ...] = ()
    loops: tuple[Loop, ...] = ()

    def agent(self, name: str) -> Agent | None:
        return next((agent for agent in self.agents if agent.name == name), None)

    def stage(self, name: str) -> Stage | None:
        return next((stage for stage in self.stages if stage.name == name), None)


@dataclass(frozen=True)
class Factoryfile:
    version: str
    factories: tuple[Factory, ...] = ()
