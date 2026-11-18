import importlib
import json
from dataclasses import fields, is_dataclass
from typing import Any, get_origin, get_type_hints

from dbos import Serializer

NAME = "agent_json"
TYPE_KEY = "__type__"
ERROR_KEY = "__error__"
OWN_MODULE = "agent_service."


class RecordedError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def encode(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {TYPE_KEY: qualified(type(value)), **{field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, BaseException):
        return {ERROR_KEY: qualified(type(value)), "message": str(value)}
    if isinstance(value, list | tuple):
        return [encode(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: encode(item) for key, item in value.items()}
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def build(name: str, raw: dict[str, Any]) -> object:
    module, _, class_name = name.rpartition(".")
    if not name.startswith(OWN_MODULE):
        raise TypeError(f"refusing to load {name}, only {OWN_MODULE} dataclasses are allowed")
    cls = getattr(importlib.import_module(module), class_name, None)
    if not isinstance(cls, type) or not is_dataclass(cls):
        raise TypeError(f"{name} is not a dataclass")
    hints = get_type_hints(cls)
    values = {}
    for field in fields(cls):
        value = decode(raw[field.name])
        values[field.name] = tuple(value) if get_origin(hints[field.name]) is tuple else value
    return cls(**values)


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if TYPE_KEY in value:
        return build(value[TYPE_KEY], value)
    if ERROR_KEY in value:
        return RecordedError(value[ERROR_KEY], value["message"])
    return {key: decode(item) for key, item in value.items()}


class AgentJsonSerializer(Serializer):
    def serialize(self, data: Any) -> str:
        return json.dumps(encode(data), ensure_ascii=False)

    def deserialize(self, serialized_data: str) -> Any:
        return decode(json.loads(serialized_data))

    def name(self) -> str:
        return NAME
