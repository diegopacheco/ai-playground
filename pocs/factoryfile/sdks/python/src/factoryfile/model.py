from dataclasses import dataclass


@dataclass(frozen=True)
class Agent:
    name: str
    description: str | None = None
    role: str | None = None
    provider: str | None = None
    model: str | None = None
    tools: tuple[str, ...] = ()
    source: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class Exhausted:
    to: str
    label: str | None = None
    reason: str | None = None
    exhausted: Exhausted | None = None


@dataclass(frozen=True)
class Route:
    outcome: str
    to: str
    label: str | None = None
    reason: str | None = None
    cycle: bool = False
    exhausted: Exhausted | None = None


@dataclass(frozen=True)
class Stage:
    name: str
    label: str | None = None
    description: str | None = None
    agents: tuple[str, ...] = ()
    run: str | None = None
    script: str | None = None
    next: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    output: str | None = None
    check: str | None = None
    verdict: dict[str, str] | None = None
    source: str | None = None
    content: str | None = None
    skills: tuple[str, ...] = ()
    tasks: str | None = None
    parallel: int | None = None
    visits: int | None = None
    end: str | None = None
    at: tuple[int, ...] = ()
    routes: tuple[Route, ...] = ()


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
    title: str | None = None
    description: str | None = None
    entry: str | None = None
    flow: str | None = None
    backlog: str | None = None
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
