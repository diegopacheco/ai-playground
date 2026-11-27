import asyncio
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote

from app import jsonx, log
from app.asgi import Receive, Scope, Send
from app.domain import supported_extensions
from app.errors import AppError, Invalid, NotFound, PayloadTooLarge
from app.service import ConversionService

logger = log.get(__name__)

_DETAIL = re.compile(r"^/api/conversions/([A-Za-z0-9_-]+)$")


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: object


async def _read_body(receive: Receive, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        part = message.get("body", b"")
        size += len(part)
        if size > limit:
            raise PayloadTooLarge(f"file is larger than {limit} bytes")
        chunks.append(part)
        if not message.get("more_body", False):
            return b"".join(chunks)


def _flag(query: dict[str, list[str]], name: str) -> bool:
    return query.get(name, ["false"])[0].lower() in ("1", "true", "yes")


class Api:
    def __init__(self, service: ConversionService, max_upload_bytes: int) -> None:
        self._service = service
        self._limit = max_upload_bytes

    async def route(self, req: Request) -> Response:
        if req.method == "GET" and req.path == "/api/health":
            return Response(200, {"status": "ok"})
        if req.method == "GET" and req.path == "/api/formats":
            return Response(200, {"extensions": list(supported_extensions())})
        if req.path == "/api/conversions":
            if req.method == "GET":
                return Response(200, self._service.history())
            if req.method == "POST":
                filename = unquote(req.headers.get("x-filename", ""))
                if not filename:
                    raise Invalid("x-filename header is required")
                ocr = _flag(req.query, "ocr")
                conversion = await asyncio.to_thread(self._service.convert, filename, req.body, ocr)
                return Response(201, conversion)
        match = _DETAIL.match(req.path)
        if match and req.method == "GET":
            return Response(200, self._service.get(match.group(1)))
        raise NotFound(f"no route for {req.method} {req.path}")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        cid, token = log.bind(headers.get(log.HEADER), scope.get("path", ""))
        try:
            try:
                req = Request(
                    method=scope["method"],
                    path=scope["path"],
                    query=parse_qs(scope.get("query_string", b"").decode()),
                    headers=headers,
                    body=await _read_body(receive, self._limit),
                )
                res = await self.route(req)
            except AppError as e:
                logger.warning("request failed status=%s error=%s", e.status, e)
                res = Response(e.status, {"error": str(e)})
            await _send(send, res, cid)
        finally:
            log.unbind(token)


async def _send(send: Send, res: Response, cid: str) -> None:
    payload = jsonx.dumps(res.body)
    await send(
        {
            "type": "http.response.start",
            "status": res.status,
            "headers": [
                (b"content-type", b"application/json"),
                (log.HEADER.encode(), cid.encode()),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})
