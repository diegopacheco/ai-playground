from common.http import Authenticator, HttpError, Request, Response, Router, asgi_app, health_handler, json_response
from common.secrets import load_signing_key
from common.settings import Settings, load_settings

from state.store import StateStore


def build_router(settings: Settings) -> Router:
    auth = Authenticator(load_signing_key(settings.secrets_dir), settings.issuer)
    store = StateStore(settings.database_url)

    async def list_keys(request: Request) -> Response:
        claims = auth.require(request, "state:read")
        entries = await store.list_prefix(claims.sub, request.query.get("prefix", ""))
        return json_response({"tenant": claims.sub, "items": [e.to_json() for e in entries]})

    async def get_key(request: Request) -> Response:
        claims = auth.require(request, "state:read")
        entry = await store.get(claims.sub, request.params["key"])
        if entry is None:
            raise HttpError(404, "not_found", "key not found")
        return json_response(entry.to_json())

    async def put_key(request: Request) -> Response:
        claims = auth.require(request, "state:write")
        body = request.json()
        if not isinstance(body, dict) or "value" not in body:
            raise HttpError(400, "invalid_request", 'body must be {"value": <json>}')
        entry = await store.put(claims.sub, request.params["key"], body["value"])
        return json_response(entry.to_json())

    async def delete_key(request: Request) -> Response:
        claims = auth.require(request, "state:write")
        if not await store.delete(claims.sub, request.params["key"]):
            raise HttpError(404, "not_found", "key not found")
        return json_response({"deleted": request.params["key"]})

    async def summary(request: Request) -> Response:
        claims = auth.require(request, "state:read:all")
        return json_response({"items": await store.summary(claims.sub)})

    async def list_all(request: Request) -> Response:
        claims = auth.require(request, "state:read:all")
        tenant = request.query.get("tenant") or None
        items = await store.list_all(claims.sub, tenant, request.query.get("prefix", ""))
        return json_response({"tenant": tenant or "all", "items": items})

    router = Router()
    router.add("GET", "/health", health_handler("state"))
    router.add("GET", "/v1/admin/state", list_all)
    router.add("GET", "/v1/admin/state/summary", summary)
    router.add("GET", "/v1/state", list_keys)
    router.add("GET", "/v1/state/{key:path}", get_key)
    router.add("PUT", "/v1/state/{key:path}", put_key)
    router.add("DELETE", "/v1/state/{key:path}", delete_key)
    return router


app = asgi_app(build_router(load_settings()))
