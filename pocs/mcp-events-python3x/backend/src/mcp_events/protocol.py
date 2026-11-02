import logging

from mcp_events.errors import METHOD_NOT_FOUND, RpcError, unsupported_delivery
from mcp_events.events_service import EventsService
from mcp_events.json_types import JsonObject
from mcp_events.params import Params
from mcp_events.sessions import Session, Sessions
from mcp_events.tools import Tools

SUPPORTED_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
SERVER_INFO: JsonObject = {"name": "mcp-events", "version": "0.1.0"}
INSTRUCTIONS = (
    "This server implements the draft MCP Events extension (events/list, events/poll, events/stream). "
    "If your host does not speak events/* use the bridge tools: call deploy_service, keep the cursor it "
    "returns, then call poll_events with that cursor and waitMs until the deployment reaches deployed or failed."
)

log = logging.getLogger("mcp_events")


class Protocol:
    def __init__(self, events: EventsService, tools: Tools, sessions: Sessions) -> None:
        self.events = events
        self._tools = tools
        self.sessions = sessions

    def initialize(self, params: Params) -> tuple[JsonObject, Session]:
        requested = params.optional_str("protocolVersion") or SUPPORTED_VERSIONS[0]
        version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        info = params.object("clientInfo")
        session = self.sessions.create(str(info.get("name", "unknown-client")), str(info.get("version", "")), version)
        log.info("initialize client=%s version=%s protocol=%s", session.client_name, session.client_version, version)
        result: JsonObject = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}, "events": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        }
        return result, session

    async def handle(self, method: str, params: Params, client_name: str) -> JsonObject:
        log.info("%s client=%s", method, client_name)
        match method:
            case "ping":
                return {}
            case "tools/list":
                return {"tools": self._tools.definitions()}
            case "tools/call":
                return await self._tools.call(params.required_str("name"), Params(params.object("arguments")), client_name)
            case "events/list":
                return self.events.list_types()
            case "events/poll":
                return self.events.poll(params)
            case "events/subscribe" | "events/unsubscribe":
                self.events.subscription(params, "webhook")
                raise unsupported_delivery("webhook")
            case _:
                raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
