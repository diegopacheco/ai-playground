import re
from collections.abc import Callable
from dataclasses import dataclass

from factoryfile.errors import FactoryfileError
from factoryfile.model import Agent, Exhausted, Factory, Factoryfile, Loop, Route, Stage
from factoryfile.parser import Attribute, Block, ListValue, MapValue, Statement, StringValue, Value

type ValueType = type[StringValue] | type[ListValue] | type[MapValue]
type Kinds = dict[str, ValueType]
type Children = dict[str, Callable[[Block], None]]
type Attrs = dict[str, Value]

KIND_NAMES: dict[ValueType, str] = {StringValue: "a string", ListValue: "a list", MapValue: "a map"}

FILE: Kinds = {"version": StringValue}
FACTORY: Kinds = {
    "title": StringValue,
    "description": StringValue,
    "entry": StringValue,
    "flow": StringValue,
    "backlog": StringValue,
}
AGENT: Kinds = {
    "description": StringValue,
    "role": StringValue,
    "provider": StringValue,
    "model": StringValue,
    "tools": ListValue,
    "source": StringValue,
    "content": StringValue,
}
STAGE: Kinds = {
    "label": StringValue,
    "description": StringValue,
    "agents": ListValue,
    "run": StringValue,
    "script": StringValue,
    "next": ListValue,
    "artifacts": ListValue,
    "output": StringValue,
    "check": StringValue,
    "verdict": MapValue,
    "source": StringValue,
    "content": StringValue,
    "skills": ListValue,
    "tasks": StringValue,
    "parallel": StringValue,
    "visits": StringValue,
    "end": StringValue,
    "at": ListValue,
}
ON: Kinds = {"to": StringValue, "label": StringValue, "reason": StringValue, "cycle": StringValue}
EXHAUSTED: Kinds = {"label": StringValue, "reason": StringValue}
LOOP: Kinds = {
    "description": StringValue,
    "from": StringValue,
    "to": StringValue,
    "until": StringValue,
    "max": StringValue,
}

FACTORY_KINDS = ("agent", "stage", "loop")

REFERENCES = {
    ("factory", "entry"): "stage",
    ("stage", "agents"): "agent",
    ("stage", "next"): "stage",
    ("stage", "tasks"): "stage",
    ("on", "to"): "stage",
    ("loop", "from"): "stage",
    ("loop", "to"): "stage",
}

NUMBERS = {("loop", "max"), ("stage", "parallel"), ("stage", "visits")}
VERDICT_KEYS = {"field", "yes", "no", "list", "notes"}

MAX_PATTERN = re.compile(r"[1-9][0-9]{0,5}")
AT_PATTERN = re.compile(r"0|[1-9][0-9]{0,2}")


@dataclass(frozen=True)
class _Reference:
    target: str
    names: tuple[str, ...]
    owner: str
    line: int
    col: int


def is_valid_max(value: str) -> bool:
    return MAX_PATTERN.fullmatch(value) is not None


def _check_attribute(attribute: Attribute, ctx: str, kinds: Kinds, attrs: Attrs) -> None:
    if attribute.name not in kinds:
        raise FactoryfileError(attribute.line, attribute.col, f'unknown attribute "{attribute.name}" in {ctx}')
    if attribute.name in attrs:
        raise FactoryfileError(attribute.line, attribute.col, f'duplicate attribute "{attribute.name}" in {ctx}')
    expected = kinds[attribute.name]
    if not isinstance(attribute.value, expected):
        raise FactoryfileError(
            attribute.line, attribute.col, f'attribute "{attribute.name}" must be {KIND_NAMES[expected]}'
        )


def _check_string(key: tuple[str, str], value: StringValue) -> None:
    name = key[1]
    if key == ("file", "version") and value.value != "1":
        raise FactoryfileError(value.line, value.col, f'unsupported version "{value.value}"')
    if key in NUMBERS and not is_valid_max(value.value):
        raise FactoryfileError(
            value.line, value.col, f'invalid {name} "{value.value}", expected an integer from 1 to 999999'
        )
    if key == ("on", "cycle") and value.value not in ("true", "false"):
        raise FactoryfileError(value.line, value.col, f'invalid cycle "{value.value}", expected "true" or "false"')


def _check_at(value: ListValue) -> None:
    if len(value.items) != 2 or not all(AT_PATTERN.fullmatch(item) for item in value.items):
        raise FactoryfileError(value.line, value.col, "invalid at, expected two integers from 0 to 999")


def _check_verdict(value: MapValue) -> None:
    for key in sorted(value.entries):
        if key not in VERDICT_KEYS:
            raise FactoryfileError(value.line, value.col, f'unknown key "{key}" in verdict')
    if "field" not in value.entries:
        raise FactoryfileError(value.line, value.col, 'verdict requires "field"')


def _check_value(ctx: str, attribute: Attribute) -> None:
    key = (ctx, attribute.name)
    value = attribute.value
    if isinstance(value, StringValue):
        _check_string(key, value)
    elif key == ("stage", "at") and isinstance(value, ListValue):
        _check_at(value)
    elif key == ("stage", "verdict") and isinstance(value, MapValue):
        _check_verdict(value)


def _collect_reference(ctx: str, owner: str, attribute: Attribute, refs: list[_Reference]) -> None:
    target = REFERENCES.get((ctx, attribute.name))
    value = attribute.value
    if target is None or isinstance(value, MapValue):
        return
    names = (value.value,) if isinstance(value, StringValue) else value.items
    refs.append(_Reference(target, names, owner, value.line, value.col))


def _check_references(refs: list[_Reference], defined: set[tuple[str, str]]) -> None:
    for ref in refs:
        for name in ref.names:
            if (ref.target, name) not in defined:
                raise FactoryfileError(ref.line, ref.col, f'unknown {ref.target} "{name}" in {ref.owner}')


def _scan(
    statements: tuple[Statement, ...],
    ctx: str,
    owner: str,
    kinds: Kinds,
    children: Children,
    refs: list[_Reference],
) -> Attrs:
    attrs: Attrs = {}
    for statement in statements:
        if isinstance(statement, Attribute):
            _check_attribute(statement, ctx, kinds, attrs)
            _check_value(ctx, statement)
            _collect_reference(ctx, owner, statement, refs)
            attrs[statement.name] = statement.value
        else:
            child = children.get(statement.kind)
            if child is None:
                raise FactoryfileError(statement.line, statement.col, f'unknown block "{statement.kind}" in {ctx}')
            child(statement)
    return attrs


def _string(attrs: Attrs, name: str) -> str | None:
    value = attrs.get(name)
    return value.value if isinstance(value, StringValue) else None


def _list(attrs: Attrs, name: str) -> tuple[str, ...]:
    value = attrs.get(name)
    return value.items if isinstance(value, ListValue) else ()


def _owner(block: Block) -> str:
    return f'{block.kind} "{block.label}"'


def _leaf(block: Block, kinds: Kinds, refs: list[_Reference]) -> Attrs:
    return _scan(block.body, block.kind, _owner(block), kinds, {}, refs)


def _require(block: Block, attrs: Attrs, name: str) -> None:
    if name not in attrs:
        raise FactoryfileError(block.line, block.col, f'{block.kind} "{block.label}" requires "{name}"')


def _agent(block: Block, refs: list[_Reference]) -> Agent:
    attrs = _leaf(block, AGENT, refs)
    has_source, has_content = "source" in attrs, "content" in attrs
    if not has_source and not has_content:
        raise FactoryfileError(block.line, block.col, f'agent "{block.label}" requires "source" or "content"')
    if has_source and has_content:
        raise FactoryfileError(
            block.line, block.col, f'agent "{block.label}" cannot set both "source" and "content"'
        )
    return Agent(
        block.label,
        _string(attrs, "description"),
        _string(attrs, "role"),
        _string(attrs, "provider"),
        _string(attrs, "model"),
        _list(attrs, "tools"),
        _string(attrs, "source"),
        _string(attrs, "content"),
    )


def _number(attrs: Attrs, name: str) -> int | None:
    value = _string(attrs, name)
    return None if value is None else int(value)


def _verdict(attrs: Attrs) -> dict[str, str] | None:
    value = attrs.get("verdict")
    return value.entries if isinstance(value, MapValue) else None


def _exhausted_child(parent: Block, owner: str, refs: list[_Reference], found: list[Exhausted]) -> Children:
    def child(item: Block) -> None:
        if found:
            raise FactoryfileError(item.line, item.col, f'duplicate exhausted in {_owner(parent)}')
        refs.append(_Reference("stage", (item.label,), owner, item.line, item.col))
        found.append(_exhausted(item, owner, refs))

    return {"exhausted": child}


def _exhausted(block: Block, owner: str, refs: list[_Reference]) -> Exhausted:
    found: list[Exhausted] = []
    attrs = _scan(block.body, "exhausted", owner, EXHAUSTED, _exhausted_child(block, owner, refs, found), refs)
    return Exhausted(
        block.label,
        _string(attrs, "label"),
        _string(attrs, "reason"),
        found[0] if found else None,
    )


def _route(block: Block, owner: str, refs: list[_Reference]) -> Route:
    found: list[Exhausted] = []
    attrs = _scan(block.body, "on", owner, ON, _exhausted_child(block, owner, refs, found), refs)
    _require(block, attrs, "to")
    return Route(
        block.label,
        _string(attrs, "to") or "",
        _string(attrs, "label"),
        _string(attrs, "reason"),
        _string(attrs, "cycle") == "true",
        found[0] if found else None,
    )


def _check_stage(block: Block, attrs: Attrs, outcomes: list[str]) -> None:
    if "agents" not in attrs and "run" not in attrs:
        raise FactoryfileError(block.line, block.col, f'stage "{block.label}" requires "agents" or "run"')
    if "source" in attrs and "content" in attrs:
        raise FactoryfileError(
            block.line, block.col, f'stage "{block.label}" cannot set both "source" and "content"'
        )
    if "next" in attrs and "next" in outcomes:
        raise FactoryfileError(
            block.line, block.col, f'stage "{block.label}" cannot set both "next" and on "next"'
        )
    if "next" in attrs or "next" in outcomes:
        for outcome in outcomes:
            if outcome != "next":
                raise FactoryfileError(
                    block.line, block.col, f'stage "{block.label}" cannot mix the next route with on "{outcome}"'
                )


def _stage(block: Block, refs: list[_Reference]) -> Stage:
    owner = _owner(block)
    routes: list[Route] = []
    outcomes: list[str] = []

    def on(item: Block) -> None:
        if item.label in outcomes:
            raise FactoryfileError(item.line, item.col, f'duplicate on "{item.label}" in {owner}')
        outcomes.append(item.label)
        routes.append(_route(item, owner, refs))

    attrs = _scan(block.body, "stage", owner, STAGE, {"on": on}, refs)
    _check_stage(block, attrs, outcomes)
    return Stage(
        block.label,
        label=_string(attrs, "label"),
        description=_string(attrs, "description"),
        agents=_list(attrs, "agents"),
        run=_string(attrs, "run"),
        script=_string(attrs, "script"),
        next=_list(attrs, "next"),
        artifacts=_list(attrs, "artifacts"),
        output=_string(attrs, "output"),
        check=_string(attrs, "check"),
        verdict=_verdict(attrs),
        source=_string(attrs, "source"),
        content=_string(attrs, "content"),
        skills=_list(attrs, "skills"),
        tasks=_string(attrs, "tasks"),
        parallel=_number(attrs, "parallel"),
        visits=_number(attrs, "visits"),
        end=_string(attrs, "end"),
        at=tuple(int(item) for item in _list(attrs, "at")),
        routes=tuple(routes),
    )


def _loop(block: Block, refs: list[_Reference]) -> Loop:
    attrs = _leaf(block, LOOP, refs)
    _require(block, attrs, "from")
    _require(block, attrs, "to")
    limit = _string(attrs, "max")
    return Loop(
        block.label,
        _string(attrs, "from") or "",
        _string(attrs, "to") or "",
        _string(attrs, "description"),
        _string(attrs, "until"),
        None if limit is None else int(limit),
    )


def _factory(block: Block) -> Factory:
    refs: list[_Reference] = []
    seen: set[tuple[str, str]] = set()
    agents: list[Agent] = []
    stages: list[Stage] = []
    loops: list[Loop] = []

    def child(item: Block) -> None:
        if (item.kind, item.label) in seen:
            raise FactoryfileError(
                item.line, item.col, f'duplicate {item.kind} "{item.label}" in factory "{block.label}"'
            )
        seen.add((item.kind, item.label))
        match item.kind:
            case "agent":
                agents.append(_agent(item, refs))
            case "stage":
                stages.append(_stage(item, refs))
            case _:
                loops.append(_loop(item, refs))

    attrs = _scan(block.body, "factory", _owner(block), FACTORY, {kind: child for kind in FACTORY_KINDS}, refs)
    _check_references(refs, seen)
    entry = _string(attrs, "entry")
    if entry is None and stages:
        entry = stages[0].name
    return Factory(
        block.label,
        _string(attrs, "title"),
        _string(attrs, "description"),
        entry,
        _string(attrs, "flow"),
        _string(attrs, "backlog"),
        tuple(agents),
        tuple(stages),
        tuple(loops),
    )


def build(statements: tuple[Statement, ...]) -> Factoryfile:
    factories: list[Factory] = []
    seen: set[str] = set()

    def factory(block: Block) -> None:
        if block.label in seen:
            raise FactoryfileError(block.line, block.col, f'duplicate factory "{block.label}"')
        seen.add(block.label)
        factories.append(_factory(block))

    attrs = _scan(statements, "file", "", FILE, {"factory": factory}, [])
    version = _string(attrs, "version")
    if version is None:
        raise FactoryfileError(1, 1, 'missing required attribute "version"')
    return Factoryfile(version, tuple(factories))
