from collections.abc import Callable
from dataclasses import dataclass, field

from agentsfile.errors import AgentsfileError
from agentsfile.model import Agent, Agentsfile, Command, Mcp, Plugin, Profile, Skill, SkillFile, Transport
from agentsfile.parser import Attribute, Block, ListValue, MapValue, Statement, StringValue, Value

type ValueType = type[StringValue] | type[ListValue] | type[MapValue]
type Kinds = dict[str, ValueType]
type Checks = dict[str, Callable[[StringValue], None]]
type Children = dict[str, Callable[[Block], None]]
type Attrs = dict[str, Value]

KIND_NAMES: dict[ValueType, str] = {StringValue: "a string", ListValue: "a list", MapValue: "a map"}

FILE: Kinds = {"version": StringValue}
PROFILE: Kinds = {"description": StringValue}
SKILL: Kinds = {"description": StringValue, "source": StringValue, "content": StringValue}
SKILL_FILE: Kinds = {"source": StringValue, "content": StringValue}
AGENT: Kinds = {
    "description": StringValue,
    "model": StringValue,
    "tools": ListValue,
    "source": StringValue,
    "content": StringValue,
}
COMMAND: Kinds = {
    "description": StringValue,
    "argument-hint": StringValue,
    "model": StringValue,
    "tools": ListValue,
    "source": StringValue,
    "content": StringValue,
}
MCP: Kinds = {
    "description": StringValue,
    "source": StringValue,
    "transport": StringValue,
    "command": StringValue,
    "args": ListValue,
    "env": MapValue,
    "url": StringValue,
    "headers": MapValue,
}
PLUGIN: Kinds = {"description": StringValue, "version": StringValue, "author": StringValue, "source": StringValue}

PROFILE_KINDS = ("skill", "agent", "command", "mcp", "plugin")
PLUGIN_KINDS = ("skill", "agent", "command", "mcp")

TRANSPORT_REQUIRES = {Transport.STDIO: "command", Transport.HTTP: "url", Transport.SSE: "url"}


@dataclass
class _Items:
    skills: list[Skill] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    mcps: list[Mcp] = field(default_factory=list)
    plugins: list[Plugin] = field(default_factory=list)

    def add(self, block: Block) -> None:
        match block.kind:
            case "skill":
                self.skills.append(_skill(block))
            case "agent":
                self.agents.append(_agent(block))
            case "command":
                self.commands.append(_command(block))
            case "mcp":
                self.mcps.append(_mcp(block))
            case _:
                self.plugins.append(_plugin(block))

    def empty(self) -> bool:
        return not (self.skills or self.agents or self.commands or self.mcps or self.plugins)


def _check_attribute(attribute: Attribute, ctx: str, kinds: Kinds, attrs: Attrs) -> None:
    if attribute.name not in kinds:
        raise AgentsfileError(attribute.line, attribute.col, f'unknown attribute "{attribute.name}" in {ctx}')
    if attribute.name in attrs:
        raise AgentsfileError(attribute.line, attribute.col, f'duplicate attribute "{attribute.name}" in {ctx}')
    expected = kinds[attribute.name]
    if not isinstance(attribute.value, expected):
        raise AgentsfileError(
            attribute.line, attribute.col, f'attribute "{attribute.name}" must be {KIND_NAMES[expected]}'
        )


def _scan(statements: tuple[Statement, ...], ctx: str, kinds: Kinds, children: Children, checks: Checks) -> Attrs:
    attrs: Attrs = {}
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


def _string(attrs: Attrs, name: str) -> str | None:
    value = attrs.get(name)
    return value.value if isinstance(value, StringValue) else None


def _list(attrs: Attrs, name: str) -> tuple[str, ...]:
    value = attrs.get(name)
    return value.items if isinstance(value, ListValue) else ()


def _map(attrs: Attrs, name: str) -> dict[str, str]:
    value = attrs.get(name)
    return dict(value.entries) if isinstance(value, MapValue) else {}


def _check_transport(value: StringValue) -> None:
    if value.value not in Transport:
        raise AgentsfileError(value.line, value.col, f'invalid transport "{value.value}", expected stdio, http or sse')


def _check_version(value: StringValue) -> None:
    if value.value != "1":
        raise AgentsfileError(value.line, value.col, f'unsupported version "{value.value}"')


def _exactly_one(block: Block, has_source: bool, has_inline: bool, inline: str) -> None:
    if not has_source and not has_inline:
        raise AgentsfileError(block.line, block.col, f'{block.kind} "{block.label}" requires "source" or {inline}')
    if has_source and has_inline:
        raise AgentsfileError(
            block.line, block.col, f'{block.kind} "{block.label}" cannot set both "source" and {inline}'
        )


def _source_or_content(block: Block, attrs: Attrs) -> None:
    _exactly_one(block, "source" in attrs, "content" in attrs, '"content"')


def _owned(owner: Block, kinds: tuple[str, ...], add: Callable[[Block], None]) -> Children:
    seen: set[tuple[str, str]] = set()

    def child(block: Block) -> None:
        if (block.kind, block.label) in seen:
            raise AgentsfileError(
                block.line, block.col, f'duplicate {block.kind} "{block.label}" in {owner.kind} "{owner.label}"'
            )
        seen.add((block.kind, block.label))
        add(block)

    return {kind: child for kind in kinds}


def _leaf(block: Block, kinds: Kinds, checks: Checks) -> Attrs:
    return _scan(block.body, block.kind, kinds, {}, checks)


def _skill_file(block: Block) -> SkillFile:
    attrs = _leaf(block, SKILL_FILE, {})
    _source_or_content(block, attrs)
    return SkillFile(block.label, _string(attrs, "source"), _string(attrs, "content"))


def _skill(block: Block) -> Skill:
    files: list[SkillFile] = []
    children = _owned(block, ("file",), lambda child: files.append(_skill_file(child)))
    attrs = _scan(block.body, "skill", SKILL, children, {})
    _source_or_content(block, attrs)
    return Skill(
        block.label, _string(attrs, "description"), _string(attrs, "source"), _string(attrs, "content"), tuple(files)
    )


def _agent(block: Block) -> Agent:
    attrs = _leaf(block, AGENT, {})
    _source_or_content(block, attrs)
    return Agent(
        block.label,
        _string(attrs, "description"),
        _string(attrs, "model"),
        _list(attrs, "tools"),
        _string(attrs, "source"),
        _string(attrs, "content"),
    )


def _command(block: Block) -> Command:
    attrs = _leaf(block, COMMAND, {})
    _source_or_content(block, attrs)
    return Command(
        block.label,
        _string(attrs, "description"),
        _string(attrs, "argument-hint"),
        _string(attrs, "model"),
        _list(attrs, "tools"),
        _string(attrs, "source"),
        _string(attrs, "content"),
    )


def _mcp(block: Block) -> Mcp:
    attrs = _leaf(block, MCP, {"transport": _check_transport})
    _exactly_one(block, "source" in attrs, "transport" in attrs, '"transport"')
    name = _string(attrs, "transport")
    transport = None if name is None else Transport(name)
    if transport is not None:
        needed = TRANSPORT_REQUIRES[transport]
        if needed not in attrs:
            raise AgentsfileError(
                block.line, block.col, f'mcp "{block.label}" with {transport} transport requires "{needed}"'
            )
    return Mcp(
        block.label,
        _string(attrs, "description"),
        _string(attrs, "source"),
        transport,
        _string(attrs, "command"),
        _list(attrs, "args"),
        _map(attrs, "env"),
        _string(attrs, "url"),
        _map(attrs, "headers"),
    )


def _plugin(block: Block) -> Plugin:
    items = _Items()
    attrs = _scan(block.body, "plugin", PLUGIN, _owned(block, PLUGIN_KINDS, items.add), {})
    _exactly_one(block, "source" in attrs, not items.empty(), "inline items")
    return Plugin(
        block.label,
        _string(attrs, "description"),
        _string(attrs, "version"),
        _string(attrs, "author"),
        _string(attrs, "source"),
        tuple(items.skills),
        tuple(items.agents),
        tuple(items.commands),
        tuple(items.mcps),
    )


def _profile(block: Block) -> Profile:
    items = _Items()
    attrs = _scan(block.body, "profile", PROFILE, _owned(block, PROFILE_KINDS, items.add), {})
    return Profile(
        block.label,
        _string(attrs, "description"),
        tuple(items.skills),
        tuple(items.agents),
        tuple(items.commands),
        tuple(items.mcps),
        tuple(items.plugins),
    )


def build(statements: tuple[Statement, ...]) -> Agentsfile:
    profiles: list[Profile] = []
    seen: set[str] = set()

    def profile(block: Block) -> None:
        if block.label in seen:
            raise AgentsfileError(block.line, block.col, f'duplicate profile "{block.label}"')
        seen.add(block.label)
        profiles.append(_profile(block))

    attrs = _scan(statements, "file", FILE, {"profile": profile}, {"version": _check_version})
    version = _string(attrs, "version")
    if version is None:
        raise AgentsfileError(1, 1, 'missing required attribute "version"')
    return Agentsfile(version, tuple(profiles))
