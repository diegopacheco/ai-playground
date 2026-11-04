import json
import subprocess
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-06-18"


class McpError(Exception):
    pass


class StdioClient:
    def __init__(self, run_script: Path, name: str) -> None:
        self.name = name
        self.process = subprocess.Popen(
            ["bash", str(run_script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.next_id = 0
        self.tools: list[str] = []

    def _send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.process.stdout is not None
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        line = self.process.stdout.readline()
        if not line:
            raise McpError(f"{self.name} closed its stdout")
        reply = json.loads(line)
        if "error" in reply:
            raise McpError(f"{self.name} {method}: {reply['error']['message']}")
        return reply["result"]

    def open(self) -> "StdioClient":
        self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "traffic", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = [tool["name"] for tool in self.request("tools/list")["tools"]]
        return self

    def call(self, tool: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        result = self.request("tools/call", {"name": tool, "arguments": arguments})
        return not result["isError"], result.get("structuredContent", result["content"][0]["text"])

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=10)
        if self.process.stdout is not None:
            self.process.stdout.close()
