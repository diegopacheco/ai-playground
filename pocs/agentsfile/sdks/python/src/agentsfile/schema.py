from collections.abc import Callable

from agentsfile.errors import AgentsfileError
from agentsfile.model import Agent, Agentsfile, Command, Mcp, Profile, Skill, Transport
from agentsfile.parser import Attribute, Block, ListValue, MapValue, Statement, StringValue, Value

type ValueType = type[StringValue] | type[ListValue] | type[MapValue]
type Kinds = dict[str, ValueType]
type Checks = dict[str, Callable[[StringValue], None]]
type Children = dict[str, Callable[[Block], None]]

KIND_NAMES: dict[ValueType, str] = {StringValue: "a string", ListValue: "a list", MapValue: "a map"}

SKILL: Kinds = {"source": StringValue, "description": StringValue}
AGENT: Kinds = {"source": StringValue, "description": StringValue, "model": StringValue, "tools": ListValue}
COMMAND: Kinds = {"source": StringValue, "description": StringValue}
MCP: Kinds = {
    "transport": StringValue,
    "description": StringValue,
    "command": StringValue,
    "args": ListValue,
    "env": MapValue,
    "url": StringValue,
    "headers": MapValue,
}
PROFILE: Kinds = {"description": StringValue}
FILE: Kinds = {"version": StringValue}

TRANSPORT_REQUIRES = {Transport.STDIO: "command", Transport.HTTP: "url", Transport.SSE: "url"}


def _check_attribute(attribute: Attribute, ctx: str, kinds: Kinds, attrs: dict[str, Value]) -> None:
    if attribute.name not in kinds:
        raise AgentsfileError(attribute.line, attribute.col, f'unknown attribute "{attribute.name}" in {ctx}')
    if attribute.name in attrs:
        raise AgentsfileError(attribute.line, attribute.col, f'duplicate attribute "{attribute.name}" in {ctx}')
    expected = kinds[attribute.name]
    if not isinstance(attribute.value, expected):
        raise AgentsfileError(
            attribute.line, attribute.col, f'attribute "{attribute.name}" must be {KIND_NAMES[expected]}'
        )


def _scan(statements: tuple[Statement, ...], ctx: str, kinds: Kinds, children: Children, checks: Checks) -> dict[str, Value]:
    attrs: dict[str, Value] = {}
    for statement in statements:
        if isinstance(statement, Attribute):
            _check_attribute(statement, ctx, kinds, attrs)
            check = checks.get(statement.name)
            if check is not None and isinstance(statement.value, StringValue):
                check(statement.value)
            attrs[statement.name] = statement.value
        else:
            child = children.get(statement.kind)
            if child is None:
                raise AgentsfileError(statement.line, statement.col, f'unknown block "{statement.kind}" in {ctx}')
            child(statement)
    return attrs


def _string(attrs: dict[str, Value], name: str) -> str | None:
    value = attrs.get(name)
    return value.value if isinstance(value, StringValue) else None


def _list(attrs: dict[str, Value], name: str) -> tuple[str, ...]:
    value = attrs.get(name)
    return value.items if isinstance(value, ListValue) else ()


def _map(attrs: dict[str, Value], name: str) -> dict[str, str]:
    value = attrs.get(name)
    return dict(value.entries) if isinstance(value, MapValue) else {}


def _required(attrs: dict[str, Value], block: Block, name: str) -> str:
    value = _string(attrs, name)
    if value is None:
        raise AgentsfileError(block.line, block.col, f'missing required attribute "{name}" in {block.kind} "{block.label}"')
    return value


def _check_transport(value: StringValue) -> None:
    if value.value not in Transport:
        raise AgentsfileError(
            value.line, value.col, f'invalid transport "{value.value}", expected stdio, http or sse'
        )


def _check_version(value: StringValue) -> None:
    if value.value != "1":
        raise AgentsfileError(value.line, value.col, f'unsupported version "{value.value}"')


def _leaf(block: Block, kinds: Kinds, checks: Checks) -> dict[str, Value]:
    return _scan(block.body, block.kind, kinds, {}, checks)


def _skill(block: Block) -> Skill:
    attrs = _leaf(block, SKILL, {})
    return Skill(block.label, _required(attrs, block, "source"), _string(attrs, "description"))


def _agent(block: Block) -> Agent:
    attrs = _leaf(block, AGENT, {})
    return Agent(
        block.label,
        _required(attrs, block, "source"),
        _string(attrs, "description"),
        _string(attrs, "model"),
        _list(attrs, "tools"),
    )


def _command(block: Block) -> Command:
    attrs = _leaf(block, COMMAND, {})
    return Command(block.label, _required(attrs, block, "source"), _string(attrs, "description"))


def _mcp(block: Block) -> Mcp:
    attrs = _leaf(block, MCP, {"transport": _check_transport})
    transport = Transport(_required(attrs, block, "transport"))
    needed = TRANSPORT_REQUIRES[transport]
    if _string(attrs, needed) is None:
        raise AgentsfileError(
            block.line, block.col, f'mcp "{block.label}" with {transport} transport requires "{needed}"'
        )
    return Mcp(
        block.label,
        transport,
        _string(attrs, "description"),
        _string(attrs, "command"),
        _list(attrs, "args"),
        _map(attrs, "env"),
        _string(attrs, "url"),
        _map(attrs, "headers"),
    )


def _unique(block: Block, seen: set[str], message: str) -> Block:
    if block.label in seen:
        raise AgentsfileError(block.line, block.col, message)
    seen.add(block.label)
    return block


def _profile(block: Block) -> Profile:
    skills: list[Skill] = []
    agents: list[Agent] = []
    commands: list[Command] = []
    mcps: list[Mcp] = []
    seen: dict[str, set[str]] = {"skill": set(), "agent": set(), "command": set(), "mcp": set()}

    def named(child: Block) -> Block:
        message = f'duplicate {child.kind} "{child.label}" in profile "{block.label}"'
        return _unique(child, seen[child.kind], message)

    children: Children = {
        "skill": lambda child: skills.append(_skill(named(child))),
        "agent": lambda child: agents.append(_agent(named(child))),
        "command": lambda child: commands.append(_command(named(child))),
        "mcp": lambda child: mcps.append(_mcp(named(child))),
    }
    attrs = _scan(block.body, "profile", PROFILE, children, {})
    return Profile(
        block.label, _string(attrs, "description"), tuple(skills), tuple(agents), tuple(commands), tuple(mcps)
    )


def build(statements: tuple[Statement, ...]) -> Agentsfile:
    profiles: list[Profile] = []
    seen: set[str] = set()

    def profile(block: Block) -> None:
        profiles.append(_profile(_unique(block, seen, f'duplicate profile "{block.label}"')))

    attrs = _scan(statements, "file", FILE, {"profile": profile}, {"version": _check_version})
    version = _string(attrs, "version")
    if version is None:
        raise AgentsfileError(1, 1, 'missing required attribute "version"')
    return Agentsfile(version, tuple(profiles))
