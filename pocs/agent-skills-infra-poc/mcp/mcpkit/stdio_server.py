import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcpkit.agentinfra import InfraError

PROTOCOL_VERSION = "2025-06-18"


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


def call_tool(tools: dict[str, Tool], name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = tools.get(name)
    if tool is None:
        return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
    for field in tool.input_schema.get("required", []):
        if field not in arguments:
            return {"content": [{"type": "text", "text": f"missing argument {field}"}], "isError": True}
    try:
        result = tool.handler(arguments)
    except (ToolError, InfraError, ValueError, TypeError) as error:
        return {"content": [{"type": "text", "text": str(error)}], "isError": True}
    return {"content": [{"type": "text", "text": json.dumps(result)}], "structuredContent": result, "isError": False}


def handle(message: dict[str, Any], name: str, version: str, tools: dict[str, Tool]) -> dict[str, Any] | None:
    method = message.get("method")
    if "id" not in message:
        return None
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
    params = message.get("params") or {}
    match method:
        case "initialize":
            reply["result"] = {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": name, "version": version},
            }
        case "ping":
            reply["result"] = {}
        case "tools/list":
            reply["result"] = {"tools": [t.describe() for t in tools.values()]}
        case "tools/call":
            reply["result"] = call_tool(tools, str(params.get("name", "")), dict(params.get("arguments") or {}))
        case _:
            reply["error"] = {"code": -32601, "message": f"method not found: {method}"}
    return reply


def serve(name: str, version: str, tool_list: list[Tool]) -> int:
    tools = {t.name: t for t in tool_list}
    if len(sys.argv) >= 3 and sys.argv[1] == "--call":
        arguments = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        result = call_tool(tools, sys.argv[2], arguments)
        print(result["content"][0]["text"])
        return 1 if result["isError"] else 0
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(message, name, version, tools)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0
