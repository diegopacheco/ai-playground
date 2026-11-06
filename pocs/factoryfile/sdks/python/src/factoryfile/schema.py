import re
from collections.abc import Callable
from dataclasses import dataclass

from factoryfile.errors import FactoryfileError
from factoryfile.model import Agent, Factory, Factoryfile, Loop, Stage
from factoryfile.parser import Attribute, Block, ListValue, MapValue, Statement, StringValue, Value

type ValueType = type[StringValue] | type[ListValue]
type Kinds = dict[str, ValueType]
type Children = dict[str, Callable[[Block], None]]
type Attrs = dict[str, Value]

KIND_NAMES: dict[ValueType, str] = {StringValue: "a string", ListValue: "a list"}

FILE: Kinds = {"version": StringValue}
FACTORY: Kinds = {"description": StringValue, "entry": StringValue}
AGENT: Kinds = {
    "description": StringValue,
    "role": StringValue,
    "model": StringValue,
    "tools": ListValue,
    "source": StringValue,
    "content": StringValue,
}
STAGE: Kinds = {
    "description": StringValue,
    "agents": ListValue,
    "next": ListValue,
    "artifacts": ListValue,
    "content": StringValue,
}
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
    ("loop", "from"): "stage",
    ("loop", "to"): "stage",
}

MAX_PATTERN = re.compile(r"[1-9][0-9]{0,5}")


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


def _check_value(ctx: str, attribute: Attribute) -> None:
    value = attribute.value
    if not isinstance(value, StringValue):
        return
    if (ctx, attribute.name) == ("file", "version") and value.value != "1":
        raise FactoryfileError(value.line, value.col, f'unsupported version "{value.value}"')
    if (ctx, attribute.name) == ("loop", "max") and not is_valid_max(value.value):
        raise FactoryfileError(
            value.line, value.col, f'invalid max "{value.value}", expected an integer from 1 to 999999'
        )


def _collect_reference(ctx: str, label: str, attribute: Attribute, refs: list[_Reference]) -> None:
    target = REFERENCES.get((ctx, attribute.name))
    value = attribute.value
    if target is None or isinstance(value, MapValue):
        return
    names = (value.value,) if isinstance(value, StringValue) else value.items
    refs.append(_Reference(target, names, f'{ctx} "{label}"', value.line, value.col))


def _check_references(refs: list[_Reference], defined: set[tuple[str, str]]) -> None:
    for ref in refs:
        for name in ref.names:
            if (ref.target, name) not in defined:
                raise FactoryfileError(ref.line, ref.col, f'unknown {ref.target} "{name}" in {ref.owner}')


def _scan(
    statements: tuple[Statement, ...],
    ctx: str,
    label: str,
    kinds: Kinds,
    children: Children,
    refs: list[_Reference],
) -> Attrs:
    attrs: Attrs = {}
    for statement in statements:
        if isinstance(statement, Attribute):
            _check_attribute(statement, ctx, kinds, attrs)
            _check_value(ctx, statement)
            _collect_reference(ctx, label, statement, refs)
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


def _leaf(block: Block, kinds: Kinds, refs: list[_Reference]) -> Attrs:
    return _scan(block.body, block.kind, block.label, kinds, {}, refs)


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
        _string(attrs, "model"),
        _list(attrs, "tools"),
        _string(attrs, "source"),
        _string(attrs, "content"),
    )


def _stage(block: Block, refs: list[_Reference]) -> Stage:
    attrs = _leaf(block, STAGE, refs)
    _require(block, attrs, "agents")
    return Stage(
        block.label,
        _string(attrs, "description"),
        _list(attrs, "agents"),
        _list(attrs, "next"),
        _list(attrs, "artifacts"),
        _string(attrs, "content"),
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

    attrs = _scan(block.body, "factory", block.label, FACTORY, {kind: child for kind in FACTORY_KINDS}, refs)
    _check_references(refs, seen)
    entry = _string(attrs, "entry")
    if entry is None and stages:
        entry = stages[0].name
    return Factory(
        block.label,
        _string(attrs, "description"),
        entry,
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
