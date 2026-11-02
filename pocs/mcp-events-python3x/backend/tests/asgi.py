import asyncio
import json
from collections.abc import MutableMapping

from mcp_events.http_app import McpHttpApp
from mcp_events.json_types import JsonObject

type Message = MutableMapping[str, object]


class Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def json(self) -> JsonObject:
        return json.loads(self.body)


def scope(method: str, path: str, headers: dict[str, str]) -> Message:
    raw = [(k.encode(), v.encode()) for k, v in headers.items()]
    return {"type": "http", "method": method, "path": path, "headers": raw}


async def request(app: McpHttpApp, method: str, path: str, body: JsonObject | None = None, headers: dict[str, str] | None = None) -> Response:
    payload = b"" if body is None else json.dumps(body).encode()
    sent: list[Message] = []
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope(method, path, headers or {}), receive, send)
    start = sent[0]
    raw_headers = start["headers"]
    assert isinstance(raw_headers, list)
    parsed = {bytes(k).decode(): bytes(v).decode() for k, v in raw_headers}
    chunks = [m.get("body") for m in sent[1:]]
    body_bytes = b"".join(c for c in chunks if isinstance(c, bytes))
    status = start["status"]
    assert isinstance(status, int)
    return Response(status, parsed, body_bytes)


class Stream:
    def __init__(self, app: McpHttpApp, body: JsonObject, headers: dict[str, str] | None = None) -> None:
        self._frames: asyncio.Queue[JsonObject] = asyncio.Queue()
        self._disconnect = asyncio.Event()
        self._payload = json.dumps(body).encode()
        self._delivered = False
        self.status = 0
        self.task = asyncio.create_task(app(scope("POST", "/mcp", headers or {}), self._receive, self._send))

    async def _receive(self) -> Message:
        if not self._delivered:
            self._delivered = True
            return {"type": "http.request", "body": self._payload, "more_body": False}
        await self._disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            status = message["status"]
            assert isinstance(status, int)
            self.status = status
            return
        body = message.get("body", b"")
        assert isinstance(body, bytes)
        text = body.decode()
        if text.startswith("data: "):
            await self._frames.put(json.loads(text.removeprefix("data: ")))
        elif text:
            await self._frames.put(json.loads(text))

    async def next(self, timeout: float = 2.0) -> JsonObject:
        return await asyncio.wait_for(self._frames.get(), timeout)

    async def close(self) -> None:
        self._disconnect.set()
        await asyncio.wait_for(self.task, 2.0)
