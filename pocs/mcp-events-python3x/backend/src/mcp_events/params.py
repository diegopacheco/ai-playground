from mcp_events.errors import invalid_params
from mcp_events.json_types import JsonObject, JsonValue


class Params:
    def __init__(self, raw: JsonValue) -> None:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise invalid_params("params must be an object")
        self.raw: JsonObject = raw

    def required_str(self, key: str) -> str:
        value = self.raw.get(key)
        if not isinstance(value, str) or not value:
            raise invalid_params(f"{key} must be a non-empty string")
        return value

    def optional_str(self, key: str) -> str | None:
        value = self.raw.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise invalid_params(f"{key} must be a string or null")
        return value

    def optional_int(self, key: str, minimum: int, maximum: int) -> int | None:
        value = self.raw.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise invalid_params(f"{key} must be an integer between {minimum} and {maximum}")
        return value

    def optional_bool(self, key: str) -> bool:
        value = self.raw.get(key, False)
        if not isinstance(value, bool):
            raise invalid_params(f"{key} must be a boolean")
        return value

    def object(self, key: str) -> JsonObject:
        value = self.raw.get(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise invalid_params(f"{key} must be an object")
        return value
