import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

from common.jwt import Claims, TokenError, verify

type Scope = dict[str, Any]
type Receive = Callable[[], Awaitable[dict[str, Any]]]
type Send = Callable[[dict[str, Any]], Awaitable[None]]

MAX_BODY = 1_048_576


class HttpError(Exception):
    def __init__(self, status: int, error: str, description: str = "") -> None:
        super().__init__(description or error)
        self.status = status
        self.error = error
        self.description = description


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: bytes
    params: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        try:
            return json.loads(self.body or b"null")
        except ValueError as error:
            raise HttpError(400, "invalid_request", "body is not valid JSON") from error

    def form(self) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(self.body.decode()).items()}


@dataclass
class Response:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


type Handler = Callable[[Request], Awaitable[Response]]


def json_response(body: Any, status: int = 200) -> Response:
    return Response(status, body)


class Router:
    def __init__(self) -> None:
        self.routes: list[tuple[str, re.Pattern[str], Handler]] = []

    def add(self, method: str, pattern: str, handler: Handler) -> None:
        regex = re.sub(r"\{(\w+):path\}", r"(?P<\1>.+)", pattern)
        regex = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", regex)
        self.routes.append((method, re.compile(f"^{regex}$"), handler))

    def resolve(self, method: str, path: str) -> tuple[Handler, dict[str, str]]:
        allowed = False
        for route_method, regex, handler in self.routes:
            match = regex.match(path)
            if match:
                if route_method == method:
                    return handler, match.groupdict()
                allowed = True
        if allowed:
            raise HttpError(405, "method_not_allowed")
        raise HttpError(404, "not_found")


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > MAX_BODY:
            raise HttpError(413, "payload_too_large")
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks)


async def _send(send: Send, response: Response) -> None:
    payload = json.dumps(response.body).encode()
    headers = [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]
    headers += [(k.lower().encode(), v.encode()) for k, v in response.headers.items()]
    await send({"type": "http.response.start", "status": response.status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


def asgi_app(router: Router) -> Callable[[Scope, Receive, Send], Awaitable[None]]:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        try:
            body = await _read_body(receive)
            handler, params = router.resolve(scope["method"], scope["path"])
            request = Request(
                method=scope["method"],
                path=scope["path"],
                query={k: v[0] for k, v in parse_qs(scope.get("query_string", b"").decode()).items()},
                headers={k.decode().lower(): v.decode() for k, v in scope["headers"]},
                body=body,
                params=params,
            )
            response = await handler(request)
        except HttpError as error:
            body_out: dict[str, str] = {"error": error.error}
            if error.description:
                body_out["error_description"] = error.description
            headers = {"www-authenticate": "Bearer"} if error.status == 401 else {}
            response = Response(error.status, body_out, headers)
        await _send(send, response)

    return app


@dataclass(frozen=True)
class Authenticator:
    key: bytes
    issuer: str

    def principal(self, request: Request) -> Claims:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HttpError(401, "invalid_token", "missing bearer token")
        try:
            return verify(header[7:].strip(), self.key, self.issuer, int(time.time()))
        except TokenError as error:
            raise HttpError(401, "invalid_token", str(error)) from error

    def require(self, request: Request, *any_of: str) -> Claims:
        claims = self.principal(request)
        if any_of and not claims.scopes.intersection(any_of):
            raise HttpError(403, "insufficient_scope", f"requires one of: {' '.join(any_of)}")
        return claims


def health_handler(service: str) -> Handler:
    async def handler(_: Request) -> Response:
        return json_response({"status": "UP", "service": service})

    return handler
