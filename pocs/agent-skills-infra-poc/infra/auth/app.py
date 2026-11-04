from common.catalog import BY_ID
from common.http import Authenticator, HttpError, Request, Response, Router, asgi_app, health_handler, json_response
from common.secrets import load_clients, load_signing_key
from common.settings import Settings, load_settings

from auth.oauth import TokenIssuer

GRANT = "client_credentials"


def build_router(settings: Settings) -> Router:
    issuer = TokenIssuer(
        clients=load_clients(settings.secrets_dir),
        key=load_signing_key(settings.secrets_dir),
        issuer=settings.issuer,
        ttl_seconds=settings.token_ttl_seconds,
    )

    async def metadata(_: Request) -> Response:
        return json_response(
            {
                "issuer": settings.issuer,
                "token_endpoint": f"{settings.issuer}/oauth/token",
                "introspection_endpoint": f"{settings.issuer}/oauth/introspect",
                "grant_types_supported": [GRANT],
                "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
                "scopes_supported": sorted({s for c in issuer.clients.values() for s in c.scopes}),
            }
        )

    async def token(request: Request) -> Response:
        form = request.form()
        if form.get("grant_type") != GRANT:
            raise HttpError(400, "unsupported_grant_type", "only client_credentials is supported")
        client = issuer.authenticate_client(request, form)
        return json_response(issuer.issue(client, form.get("scope", "")))

    async def introspect(request: Request) -> Response:
        form = request.form()
        issuer.authenticate_client(request, form)
        if "token" not in form:
            raise HttpError(400, "invalid_request", "token is required")
        return json_response(issuer.introspect(form["token"]))

    guard = Authenticator(issuer.key, settings.issuer)

    async def list_clients(request: Request) -> Response:
        guard.require(request, "clients:read")
        items = []
        for client in issuer.clients.values():
            entry = BY_ID.get(client.client_id)
            items.append(
                {
                    "client_id": client.client_id,
                    "kind": entry.kind if entry else "unknown",
                    "description": entry.description if entry else "",
                    "scopes": sorted(client.scopes),
                }
            )
        return json_response({"items": items})

    async def me(request: Request) -> Response:
        claims = guard.principal(request)
        entry = BY_ID.get(claims.sub)
        return json_response(
            {
                "client_id": claims.sub,
                "kind": entry.kind if entry else "unknown",
                "description": entry.description if entry else "",
                "scopes": sorted(claims.scopes),
                "iat": claims.iat,
                "exp": claims.exp,
                "iss": claims.iss,
            }
        )

    router = Router()
    router.add("GET", "/health", health_handler("auth"))
    router.add("GET", "/v1/me", me)
    router.add("GET", "/v1/admin/clients", list_clients)
    router.add("GET", "/.well-known/oauth-authorization-server", metadata)
    router.add("POST", "/oauth/token", token)
    router.add("POST", "/oauth/introspect", introspect)
    return router


app = asgi_app(build_router(load_settings()))
