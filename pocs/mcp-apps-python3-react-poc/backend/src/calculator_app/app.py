import os

from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware

from calculator_app.server import build_server

LOCAL_HOSTS = ["localhost:*", "127.0.0.1:*"]
LOCAL_ORIGINS = ["http://localhost:*", "http://127.0.0.1:*"]
LOCAL_ORIGIN_PATTERN = r"http://(localhost|127\.0\.0\.1)(:\d+)?"
MCP_HEADERS = ["mcp-session-id", "mcp-protocol-version", "last-event-id"]


def transport_security(public_host: str) -> TransportSecuritySettings:
    hosts = list(LOCAL_HOSTS)
    origins = list(LOCAL_ORIGINS)
    if public_host:
        hosts += [public_host, f"{public_host}:*"]
        origins += [f"https://{public_host}", f"https://{public_host}:*"]
    return TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)


def create_app(public_host: str = "") -> Starlette:
    app = build_server().streamable_http_app(transport_security=transport_security(public_host))
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCAL_ORIGIN_PATTERN,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["content-type", "accept", "authorization", *MCP_HEADERS],
        expose_headers=MCP_HEADERS,
    )
    return app


app = create_app(os.environ.get("MCP_PUBLIC_HOST", ""))
