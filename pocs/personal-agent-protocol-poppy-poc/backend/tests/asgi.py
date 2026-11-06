import asyncio
import json
from collections.abc import MutableMapping

from poppy_company.app import CompanyApp
from poppy_company.json_types import JsonObject, JsonValue
from signer import BASE, SigningAgent

type Message = MutableMapping[str, object]


class Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def json(self) -> JsonObject:
        return json.loads(self.body)


async def call(app: CompanyApp, method: str, path: str, body: JsonObject | None, headers: dict[str, str]) -> Response:
    payload = b"" if body is None else json.dumps(body).encode()
    raw_path, _, query = path.partition("?")
    scope: Message = {"type": "http", "method": method, "path": raw_path, "query_string": query.encode(), "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    start = sent[0]
    raw_headers = start["headers"]
    status = start["status"]
    assert isinstance(raw_headers, list) and isinstance(status, int)
    chunk = sent[1].get("body", b"")
    assert isinstance(chunk, bytes)
    return Response(status, {bytes(k).decode(): bytes(v).decode() for k, v in raw_headers}, chunk)


def request(app: CompanyApp, method: str, path: str, body: JsonObject | None = None, headers: dict[str, str] | None = None) -> Response:
    return asyncio.run(call(app, method, path, body, headers or {}))


def signed(app: CompanyApp, agent: SigningAgent, method: str, path: str, body: JsonObject | None = None) -> Response:
    url = f"{BASE}{path.partition('?')[0]}"
    return request(app, method, path, body, {"poppy-session": agent.session(method, url), "poppy-version": "0.4"})


def new_app(agent: SigningAgent) -> CompanyApp:
    app = CompanyApp(BASE, clock=lambda: agent.now)
    app.directory.publish(agent.name, agent.metadata())
    return app


def obj(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def objs(value: JsonValue) -> list[JsonObject]:
    assert isinstance(value, list)
    return [obj(item) for item in value]
