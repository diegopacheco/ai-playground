import asyncio

from mcp_events.app import create_app
from mcp_events.http_app import McpHttpApp
from mcp_events.json_types import JsonObject, JsonValue

from asgi import Stream, request

STEP = 0.02


def rpc(method: str, params: JsonObject | None = None, request_id: int = 1) -> JsonObject:
    body: JsonObject = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


async def connect(app: McpHttpApp, name: str) -> dict[str, str]:
    response = await request(app, "POST", "/mcp", rpc("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": name, "version": "1"}}))
    return {"mcp-session-id": response.headers["mcp-session-id"]}


async def call(app: McpHttpApp, headers: dict[str, str], method: str, params: JsonObject | None = None) -> JsonObject:
    response = await request(app, "POST", "/mcp", rpc(method, params), headers)
    return response.json()


def result(body: JsonObject) -> JsonObject:
    value = body["result"]
    assert isinstance(value, dict)
    return value


def error_code(body: JsonObject) -> JsonValue:
    error = body["error"]
    assert isinstance(error, dict)
    return error["code"]


def structured(body: JsonObject) -> JsonObject:
    value = result(body)["structuredContent"]
    assert isinstance(value, dict)
    return value


def statuses(events: JsonValue) -> list[JsonValue]:
    assert isinstance(events, list)
    return [e["data"]["status"] for e in events if isinstance(e, dict) and isinstance(e["data"], dict)]


def test_initialize_advertises_events_capability_and_opens_a_session() -> None:
    async def scenario() -> None:
        app = create_app()
        response = await request(app, "POST", "/mcp", rpc("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "claude-code"}}))
        body = result(response.json())
        assert body["protocolVersion"] == "2025-06-18"
        assert body["capabilities"] == {"tools": {"listChanged": False}, "events": {"listChanged": False}}
        assert response.headers["mcp-session-id"]

    asyncio.run(scenario())


def test_poll_without_cursor_starts_from_now_then_returns_the_full_deploy_lifecycle() -> None:
    async def scenario() -> None:
        app = create_app(step_seconds=STEP)
        headers = await connect(app, "claude-code")
        first = result(await call(app, headers, "events/poll", {"name": "deploy.status_changed"}))
        assert first["events"] == []
        await call(app, headers, "tools/call", {"name": "deploy_service", "arguments": {"service": "checkout", "env": "dev"}})
        await asyncio.sleep(STEP * 5)
        second = result(await call(app, headers, "events/poll", {"name": "deploy.status_changed", "cursor": first["cursor"]}))
        assert statuses(second["events"]) == ["queued", "building", "testing", "deployed"]
        third = result(await call(app, headers, "events/poll", {"name": "deploy.status_changed", "cursor": second["cursor"]}))
        assert third["events"] == []

    asyncio.run(scenario())


def test_events_carry_the_mcp_client_that_triggered_them() -> None:
    async def scenario() -> None:
        app = create_app(step_seconds=STEP)
        headers = await connect(app, "codex-mcp-client")
        deploy = structured(await call(app, headers, "tools/call", {"name": "deploy_service", "arguments": {"service": "search", "env": "prod"}}))
        assert deploy["triggeredBy"] == "codex-mcp-client"
        polled = result(await call(app, headers, "events/poll", {"name": "deploy.status_changed", "cursor": deploy["cursor"]}))
        events = polled["events"]
        assert isinstance(events, list) and isinstance(events[0], dict) and isinstance(events[0]["data"], dict)
        assert events[0]["data"]["triggeredBy"] == "codex-mcp-client"

    asyncio.run(scenario())


def test_failed_deploy_raises_an_alert_with_severity_from_the_environment() -> None:
    async def scenario() -> None:
        app = create_app(step_seconds=STEP)
        headers = await connect(app, "ui")
        start = result(await call(app, headers, "events/poll", {"name": "alert.raised"}))
        await call(app, headers, "tools/call", {"name": "deploy_service", "arguments": {"service": "cart", "env": "prod", "simulate_failure": True}})
        await asyncio.sleep(STEP * 5)
        alerts = result(await call(app, headers, "events/poll", {"name": "alert.raised", "arguments": {"minSeverity": "P1"}, "cursor": start["cursor"]}))
        events = alerts["events"]
        assert isinstance(events, list) and len(events) == 1
        assert isinstance(events[0], dict) and isinstance(events[0]["data"], dict)
        assert events[0]["data"]["severity"] == "P1"

    asyncio.run(scenario())


def test_poll_events_tool_waits_until_an_event_arrives_so_agents_do_not_busy_loop() -> None:
    async def scenario() -> None:
        app = create_app(step_seconds=0.2)
        headers = await connect(app, "claude-code")
        deploy = structured(await call(app, headers, "tools/call", {"name": "deploy_service", "arguments": {"service": "api", "env": "dev"}}))
        first = structured(await call(app, headers, "tools/call", {"name": "poll_events", "arguments": {"name": "deploy.status_changed", "cursor": deploy["cursor"]}}))
        assert statuses(first["events"]) == ["queued"]
        waited = structured(await call(app, headers, "tools/call", {"name": "poll_events", "arguments": {"name": "deploy.status_changed", "cursor": first["cursor"], "waitMs": 2000}}))
        assert statuses(waited["events"]) == ["building"]

    asyncio.run(scenario())


def test_protocol_errors_use_the_draft_error_codes() -> None:
    async def scenario() -> None:
        app = create_app()
        headers = await connect(app, "ui")
        assert error_code(await call(app, headers, "events/poll", {"name": "email.received"})) == -32023
        assert error_code(await call(app, headers, "events/poll", {"name": "alert.raised", "arguments": {"minSeverity": "P9"}})) == -32602
        assert error_code(await call(app, headers, "events/poll", {"name": "alert.raised", "cursor": "garbage"})) == -32602
        webhook: JsonObject = {"name": "alert.raised", "delivery": {"mode": "webhook", "url": "https://hooks.local/x"}}
        assert error_code(await call(app, headers, "events/subscribe", webhook)) == -32026

    asyncio.run(scenario())


def test_push_stream_confirms_then_delivers_events_with_cursor_and_subscription_id() -> None:
    async def scenario() -> None:
        app = create_app(step_seconds=STEP)
        headers = await connect(app, "ui")
        stream = Stream(app, rpc("events/stream", {"name": "deploy.status_changed", "arguments": {"env": "staging"}}, 7), headers)
        active = await stream.next()
        assert active["method"] == "notifications/events/active"
        await call(app, headers, "tools/call", {"name": "deploy_service", "arguments": {"service": "a", "env": "dev"}})
        await call(app, headers, "tools/call", {"name": "deploy_service", "arguments": {"service": "b", "env": "staging"}})
        frames = [await stream.next() for _ in range(4)]
        assert [f["method"] for f in frames] == ["notifications/events/event"] * 4
        params = [f["params"] for f in frames]
        assert all(isinstance(p, dict) and isinstance(p["data"], dict) and p["data"]["service"] == "b" for p in params)
        assert all(isinstance(p, dict) and p["_meta"] == {"io.modelcontextprotocol/subscriptionId": 7} for p in params)
        assert stream.status == 200
        await stream.close()

    asyncio.run(scenario())


def test_push_stream_reconnect_with_cursor_replays_what_was_missed() -> None:
    async def scenario() -> None:
        app = create_app(step_seconds=STEP)
        headers = await connect(app, "ui")
        start = result(await call(app, headers, "events/poll", {"name": "deploy.status_changed"}))
        await call(app, headers, "tools/call", {"name": "deploy_service", "arguments": {"service": "a", "env": "dev"}})
        await asyncio.sleep(STEP * 5)
        stream = Stream(app, rpc("events/stream", {"name": "deploy.status_changed", "cursor": start["cursor"]}, 3), headers)
        active = await stream.next()
        assert isinstance(active["params"], dict) and active["params"]["truncated"] is False
        assert active["params"]["cursor"] == start["cursor"]
        replayed = [await stream.next() for _ in range(4)]
        assert statuses([f["params"] for f in replayed]) == ["queued", "building", "testing", "deployed"]
        await stream.close()

    asyncio.run(scenario())


def test_push_stream_sends_heartbeats_with_cursor_when_nothing_happens() -> None:
    async def scenario() -> None:
        app = create_app(heartbeat_seconds=0.05)
        headers = await connect(app, "ui")
        stream = Stream(app, rpc("events/stream", {"name": "alert.raised"}, 1), headers)
        await stream.next()
        heartbeat = await stream.next()
        assert heartbeat["method"] == "notifications/events/heartbeat"
        assert isinstance(heartbeat["params"], dict) and heartbeat["params"]["cursor"] == "c_0"
        await stream.close()

    asyncio.run(scenario())


def test_invalid_stream_request_returns_an_error_and_opens_no_stream() -> None:
    async def scenario() -> None:
        app = create_app()
        response = await request(app, "POST", "/mcp", rpc("events/stream", {"name": "nope"}))
        assert response.headers["content-type"] == "application/json"
        assert error_code(response.json()) == -32023

    asyncio.run(scenario())


def test_foreign_origin_is_rejected_to_block_dns_rebinding() -> None:
    async def scenario() -> None:
        app = create_app()
        evil = await request(app, "POST", "/mcp", rpc("ping"), {"origin": "https://evil.example"})
        local = await request(app, "POST", "/mcp", rpc("ping"), {"origin": "http://localhost:5290"})
        assert evil.status == 403
        assert local.status == 200

    asyncio.run(scenario())


def test_clients_stay_listed_after_they_disconnect_so_the_ui_shows_who_used_the_server() -> None:
    async def scenario() -> None:
        app = create_app()
        await connect(app, "claude-code")
        codex = await connect(app, "codex-mcp-client")
        await request(app, "DELETE", "/mcp", None, codex)
        clients = (await request(app, "GET", "/api/clients")).json()["clients"]
        assert isinstance(clients, list)
        state = {str(c["clientName"]): c["closed"] for c in clients if isinstance(c, dict)}
        assert state == {"claude-code": False, "codex-mcp-client": True}

    asyncio.run(scenario())
