import json
from collections.abc import Awaitable, Callable, MutableMapping
from urllib.parse import parse_qs, urlsplit

from poppy_company.errors import invalid_request
from poppy_company.json_types import JsonObject, JsonValue

type Scope = MutableMapping[str, object]
type Message = MutableMapping[str, object]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]

ALLOWED_ORIGIN_HOSTS = ("localhost", "127.0.0.1")


class Request:
    def __init__(self, scope: Scope, body: bytes) -> None:
        self.method = str(scope.get("method", "GET"))
        self.path = str(scope.get("path", "/"))
        raw_query = scope.get("query_string", b"")
        self.query = parse_qs(raw_query.decode() if isinstance(raw_query, bytes) else "")
        self.headers = read_headers(scope)
        self.body = body

    def header(self, name: str) -> str | None:
        return self.headers.get(name)

    def param(self, name: str) -> str | None:
        values = self.query.get(name)
        return values[0] if values else None

    def json(self) -> JsonObject:
        try:
            value = json.loads(self.body or b"{}")
        except ValueError as error:
            raise invalid_request("body must be JSON") from error
        if not isinstance(value, dict):
            raise invalid_request("body must be a JSON object")
        return value


def read_headers(scope: Scope) -> dict[str, str]:
    headers = scope.get("headers")
    if not isinstance(headers, list | tuple):
        return {}
    return {bytes(key).decode("latin-1").lower(): bytes(value).decode("latin-1") for key, value in headers}


async def read_body(receive: Receive) -> bytes:
    body = b""
    while True:
        message = await receive()
        chunk = message.get("body", b"")
        body += chunk if isinstance(chunk, bytes) else b""
        if message["type"] == "http.disconnect" or not message.get("more_body", False):
            return body


def cors_headers(origin: str | None) -> list[tuple[bytes, bytes]]:
    if origin is None or urlsplit(origin).hostname not in ALLOWED_ORIGIN_HOSTS:
        return []
    return [
        (b"access-control-allow-origin", origin.encode()),
        (b"access-control-allow-methods", b"GET, POST, PUT, OPTIONS"),
        (b"access-control-allow-headers", b"content-type, poppy-session, poppy-version"),
        (b"vary", b"origin"),
    ]


async def send_json(send: Send, status: int, body: JsonValue, origin: str | None) -> None:
    payload = b"" if body is None else json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), *cors_headers(origin)]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})
