from starlette.testclient import TestClient

from calculator_app.app import create_app

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe", "version": "1"}},
}
LOCAL = "http://localhost:8710"
STREAMABLE = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def test_browser_clients_on_another_local_port_pass_the_cors_preflight():
    with TestClient(create_app(), base_url=LOCAL) as browser:
        response = browser.options(
            "/mcp",
            headers={
                "origin": "http://localhost:6274",
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type,mcp-session-id,mcp-protocol-version",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:6274"


def test_session_id_is_exposed_so_browser_clients_can_keep_the_session():
    with TestClient(create_app(), base_url=LOCAL) as browser:
        response = browser.post("/mcp", json=INITIALIZE, headers={**STREAMABLE, "origin": "http://127.0.0.1:6274"})
    assert response.status_code == 200
    assert "mcp-session-id" in response.headers["access-control-expose-headers"]
    assert response.headers["mcp-session-id"]


def test_remote_origins_get_no_cors_grant():
    with TestClient(create_app(), base_url=LOCAL) as browser:
        response = browser.options(
            "/mcp",
            headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
        )
    assert "access-control-allow-origin" not in response.headers


def test_public_tunnel_host_is_accepted_when_configured_so_chatgpt_can_reach_the_server():
    with TestClient(create_app("calc.example.ts.net")) as chatgpt:
        response = chatgpt.post("/mcp", json=INITIALIZE, headers={**STREAMABLE, "host": "calc.example.ts.net:8443"})
    assert response.status_code == 200


def test_unknown_host_is_rejected_to_block_dns_rebinding():
    with TestClient(create_app()) as stranger:
        response = stranger.post("/mcp", json=INITIALIZE, headers={**STREAMABLE, "host": "calc.example.ts.net"})
    assert response.status_code == 421
