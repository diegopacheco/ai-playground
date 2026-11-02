import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from urllib.parse import urlsplit

from mcp_events.errors import INTERNAL_ERROR, INVALID_REQUEST, PARSE_ERROR, RpcError
from mcp_events.json_types import JsonObject, JsonValue
from mcp_events.params import Params
from mcp_events.protocol import Protocol
from mcp_events.sessions import UNKNOWN_CLIENT

type Scope = MutableMapping[str, object]
type Message = MutableMapping[str, object]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]

SESSION_HEADER = "mcp-session-id"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

log = logging.getLogger("mcp_events")


def header(scope: Scope, name: str) -> str | None:
    headers = scope.get("headers")
    if not isinstance(headers, list | tuple):
        return None
    for key, value in headers:
        if bytes(key).decode("latin-1").lower() == name:
            return bytes(value).decode("latin-1")
    return None


def local_origin(origin: str | None) -> bool:
    return origin is None or urlsplit(origin).hostname in LOCAL_HOSTS


def rpc_error(request_id: JsonValue, error: RpcError) -> JsonObject:
    return {"jsonrpc": "2.0", "id": request_id, "error": error.to_json()}


def rpc_result(request_id: JsonValue, result: JsonObject) -> JsonObject:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


async def read_body(receive: Receive) -> bytes:
    body = b""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return body
        chunk = message.get("body", b"")
        body += chunk if isinstance(chunk, bytes) else b""
        if not message.get("more_body", False):
            return body


async def send_json(send: Send, status: int, body: JsonValue, extra: list[tuple[bytes, bytes]] | None = None) -> None:
    payload = b"" if body is None else json.dumps(body).encode()
    headers = [(b"content-type", b"application/json")] + (extra or [])
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


async def wait_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


async def pump_sse(send: Send, messages: AsyncIterator[JsonObject]) -> None:
    async for message in messages:
        await send({"type": "http.response.body", "body": f"data: {json.dumps(message)}\n\n".encode(), "more_body": True})


class McpHttpApp:
    def __init__(self, protocol: Protocol) -> None:
        self._protocol = protocol

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        if not local_origin(header(scope, "origin")):
            await send_json(send, 403, {"error": "origin not allowed"})
            return
        route = (str(scope["method"]), str(scope["path"]).rstrip("/") or "/")
        match route:
            case ("POST", "/mcp"):
                await self._post(scope, receive, send)
            case ("GET", "/mcp"):
                await send_json(send, 405, None, [(b"allow", b"POST, DELETE")])
            case ("DELETE", "/mcp"):
                self._protocol.sessions.close(header(scope, SESSION_HEADER))
                await send_json(send, 200, {})
            case ("GET", "/health"):
                await send_json(send, 200, {"status": "ok"})
            case ("GET", "/api/clients"):
                await send_json(send, 200, {"clients": [s.to_json() for s in self._protocol.sessions.all()]})
            case _:
                await send_json(send, 404, {"error": "not found"})

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _post(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            body = json.loads(await read_body(receive))
        except ValueError:
            await send_json(send, 400, rpc_error(None, RpcError(PARSE_ERROR, "Parse error")))
            return
        if not isinstance(body, dict) or not isinstance(body.get("method"), str):
            await send_json(send, 400, rpc_error(None, RpcError(INVALID_REQUEST, "Invalid request")))
            return
        method: str = body["method"]
        if "id" not in body:
            await send_json(send, 202, None)
            return
        request_id: JsonValue = body["id"]
        session = self._protocol.sessions.touch(header(scope, SESSION_HEADER))
        client_name = session.client_name if session else UNKNOWN_CLIENT
        try:
            params = Params(body.get("params"))
            if method == "initialize":
                result, created = self._protocol.initialize(params)
                await send_json(send, 200, rpc_result(request_id, result), [(SESSION_HEADER.encode(), created.session_id.encode())])
                return
            if method == "events/stream":
                subscription = self._protocol.events.subscription(params, "push")
                log.info("events/stream client=%s name=%s", client_name, subscription.event_type.name)
                await self._stream(receive, send, self._protocol.events.stream(subscription, request_id))
                return
            await send_json(send, 200, rpc_result(request_id, await self._protocol.handle(method, params, client_name)))
        except RpcError as error:
            await send_json(send, 200, rpc_error(request_id, error))
        except Exception:
            log.exception("request failed method=%s", method)
            await send_json(send, 200, rpc_error(request_id, RpcError(INTERNAL_ERROR, "Internal error")))

    async def _stream(self, receive: Receive, send: Send, messages: AsyncIterator[JsonObject]) -> None:
        headers = [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"), (b"x-accel-buffering", b"no")]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        pump = asyncio.create_task(pump_sse(send, messages))
        disconnect = asyncio.create_task(wait_disconnect(receive))
        done, pending = await asyncio.wait({pump, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if pump in done and pump.exception() is not None:
            log.warning("stream ended: %s", pump.exception())
        log.info("events/stream closed")
