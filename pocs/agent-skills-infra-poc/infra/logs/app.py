import time

from common.http import Authenticator, HttpError, Request, Response, Router, asgi_app, health_handler, json_response
from common.secrets import load_clients, load_signing_key
from common.settings import Settings, load_settings

from logs.loki import LogLine, LokiClient, validate


def bounded_int(raw: str | None, default: int, low: int, high: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise HttpError(400, "invalid_request", f"{raw} is not an integer") from error
    return max(low, min(high, value))


def build_router(settings: Settings) -> Router:
    auth = Authenticator(load_signing_key(settings.secrets_dir), settings.issuer)
    loki = LokiClient(settings.loki_url)

    async def write(request: Request) -> Response:
        claims = auth.require(request, "logs:write")
        body = request.json()
        if not isinstance(body, dict):
            raise HttpError(400, "invalid_request", "body must be an object")
        level = str(body.get("level", "info"))
        message = str(body.get("message", ""))
        fields = body.get("fields", {})
        validate(level, message, fields)
        line = LogLine(time.time_ns(), level, message, fields)
        await loki.push(claims.sub, line)
        return json_response({"tenant": claims.sub, "accepted": line.to_json()}, 202)

    async def read(request: Request) -> Response:
        claims = auth.require(request, "logs:read")
        since = bounded_int(request.query.get("since_minutes"), 60, 1, 10_080)
        limit = bounded_int(request.query.get("limit"), 100, 1, 1000)
        lines = await loki.query(claims.sub, request.query.get("level") or None, since, limit)
        return json_response({"tenant": claims.sub, "items": [line.to_json() for line in lines]})

    tenants = sorted(load_clients(settings.secrets_dir))

    async def summary(request: Request) -> Response:
        auth.require(request, "logs:read:all")
        since = bounded_int(request.query.get("since_minutes"), 60, 1, 1440)
        limit = bounded_int(request.query.get("limit"), 40, 1, 200)
        return json_response({"since_minutes": since, **await loki.summary(tenants, since, limit)})

    async def read_all(request: Request) -> Response:
        auth.require(request, "logs:read:all")
        tenant = request.query.get("tenant") or None
        if tenant is not None and tenant not in tenants:
            raise HttpError(400, "invalid_tenant", "unknown tenant")
        since = bounded_int(request.query.get("since_minutes"), 60, 1, 10_080)
        limit = bounded_int(request.query.get("limit"), 200, 1, 1000)
        lines = await loki.query_many([tenant] if tenant else tenants, request.query.get("level") or None, since, limit)
        return json_response({"tenant": tenant or "all", "items": lines})

    router = Router()
    router.add("GET", "/health", health_handler("logs"))
    router.add("GET", "/v1/admin/logs", read_all)
    router.add("GET", "/v1/admin/logs/summary", summary)
    router.add("POST", "/v1/logs", write)
    router.add("GET", "/v1/logs", read)
    return router


app = asgi_app(build_router(load_settings()))
