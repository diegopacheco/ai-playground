import asyncio
import json

from mcp_events.catalog import ENVIRONMENTS, string_property
from mcp_events.deployer import Deployer
from mcp_events.errors import RpcError
from mcp_events.events_service import EventsService
from mcp_events.json_types import JsonObject, JsonValue
from mcp_events.params import Params

MAX_WAIT_MS = 30000

DEPLOY_SERVICE: JsonObject = {
    "name": "deploy_service",
    "description": (
        "Start a simulated deployment. It emits deploy.status_changed events (queued, building, testing, "
        "deployed or failed) and alert.raised when it fails. Returns the deployId and a cursor taken right "
        "before the first event: pass that cursor to poll_events to read every event of this deployment."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "service": string_property("Service name, for example checkout"),
            "env": string_property("Target environment", list[str](ENVIRONMENTS)),
            "simulate_failure": {"type": "boolean", "description": "Fail in testing and raise an alert", "default": False},
        },
        "required": ["service", "env"],
        "additionalProperties": False,
    },
}

LIST_EVENT_TYPES: JsonObject = {
    "name": "list_event_types",
    "description": "Bridge to the MCP events/list method. Returns every event type, its delivery modes and schemas.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
}

POLL_EVENTS: JsonObject = {
    "name": "poll_events",
    "description": (
        "Bridge to the MCP events/poll method for hosts that do not speak events/* yet. Pass the event name, "
        "optional filter arguments and the cursor from the previous call or from deploy_service. Without a "
        "cursor it starts from now and returns no events. waitMs keeps the call open until an event arrives "
        "or the wait ends. Always keep the returned cursor for the next call."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "name": string_property("Event type name, for example deploy.status_changed"),
            "arguments": {"type": "object", "description": "Filter arguments from the event type inputSchema"},
            "cursor": string_property("Cursor returned by the previous call"),
            "maxEvents": {"type": "integer", "minimum": 1, "maximum": 500},
            "waitMs": {"type": "integer", "minimum": 0, "maximum": MAX_WAIT_MS},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}


def tool_result(value: JsonObject, is_error: bool = False) -> JsonObject:
    return {
        "content": [{"type": "text", "text": json.dumps(value, indent=2)}],
        "structuredContent": value,
        "isError": is_error,
    }


class Tools:
    def __init__(self, events: EventsService, deployer: Deployer) -> None:
        self._events = events
        self._deployer = deployer

    def definitions(self) -> list[JsonValue]:
        return [DEPLOY_SERVICE, LIST_EVENT_TYPES, POLL_EVENTS]

    async def call(self, name: str, arguments: Params, client_name: str) -> JsonObject:
        try:
            match name:
                case "deploy_service":
                    return tool_result(self._deploy(arguments, client_name))
                case "list_event_types":
                    return tool_result(self._events.list_types())
                case "poll_events":
                    return tool_result(await self._poll(arguments))
                case _:
                    return tool_result({"error": f"unknown tool {name}"}, True)
        except RpcError as error:
            return tool_result({"error": error.to_json()}, True)

    def _deploy(self, arguments: Params, client_name: str) -> JsonObject:
        cursor = self._events.head_cursor
        deploy = self._deployer.start(
            arguments.required_str("service"),
            arguments.required_str("env"),
            arguments.optional_bool("simulate_failure"),
            client_name,
        )
        return {"deployId": deploy.deploy_id, "service": deploy.service, "env": deploy.env, "triggeredBy": client_name, "cursor": cursor}

    async def _poll(self, arguments: Params) -> JsonObject:
        wait_ms = arguments.optional_int("waitMs", 0, MAX_WAIT_MS) or 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_ms / 1000
        while True:
            result = self._events.poll(arguments)
            events = result.get("events")
            remaining = deadline - loop.time()
            if (isinstance(events, list) and events) or arguments.optional_str("cursor") is None or remaining <= 0:
                return result
            await self._events.wait_for_change(remaining)
