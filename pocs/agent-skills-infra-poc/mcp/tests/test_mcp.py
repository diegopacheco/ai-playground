import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from typing import Any

MCP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MCP_DIR))

from mcpkit.stdio_server import Tool, handle

BIN = str(MCP_DIR.parent / "infra" / "bin" / "agentinfra")


def echo(args: dict[str, Any]) -> dict[str, Any]:
    if args["text"] == "boom":
        raise ValueError("boom")
    return {"echo": args["text"]}


TOOLS = {"echo": Tool("echo", "echo", {"type": "object", "required": ["text"]}, echo)}


class ProtocolTest(unittest.TestCase):
    def test_initialize_echoes_the_client_protocol_version(self) -> None:
        reply = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}, "s", "1", TOOLS)
        assert reply is not None
        self.assertEqual(reply["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", reply["result"]["capabilities"])

    def test_notifications_get_no_reply(self) -> None:
        self.assertIsNone(handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, "s", "1", TOOLS))

    def test_tool_failures_are_tool_errors_not_protocol_errors_so_the_model_can_react(self) -> None:
        reply = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "boom"}}}, "s", "1", TOOLS)
        assert reply is not None
        self.assertTrue(reply["result"]["isError"])
        self.assertNotIn("error", reply)

    def test_missing_required_argument_is_reported(self) -> None:
        reply = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {}}}, "s", "1", TOOLS)
        assert reply is not None
        self.assertIn("missing argument text", reply["result"]["content"][0]["text"])

    def test_unknown_method_is_json_rpc_method_not_found(self) -> None:
        reply = handle({"jsonrpc": "2.0", "id": 4, "method": "resources/list"}, "s", "1", TOOLS)
        assert reply is not None
        self.assertEqual(reply["error"]["code"], -32601)


class StdioSession:
    def __init__(self, server: str) -> None:
        self.process = subprocess.Popen(
            ["bash", str(MCP_DIR / server / "run.sh")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.next_id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.next_id += 1
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if reply["id"] != self.next_id:
            raise AssertionError("reply id mismatch")
        return reply

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": tool, "arguments": arguments})["result"]

    def close(self) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()


def state_keys(client: str, prefix: str) -> list[str]:
    out = subprocess.run([BIN, "--client", client, "state", "list", "--prefix", prefix], capture_output=True, text=True, check=True)
    return [item["key"] for item in json.loads(out.stdout)["items"]]


class BookmarksLiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = StdioSession("bookmarks-mcp")
        self.session.request("initialize", {"protocolVersion": "2025-06-18"})
        self.url = f"https://itest.local/{uuid.uuid4().hex}"

    def tearDown(self) -> None:
        self.session.call("delete_bookmark", {"url": self.url})
        self.session.close()

    def test_tools_are_listed(self) -> None:
        names = [t["name"] for t in self.session.request("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["save_bookmark", "list_bookmarks", "search_bookmarks", "delete_bookmark"])

    def test_saved_bookmark_lives_in_the_mcp_tenant_only(self) -> None:
        saved = self.session.call("save_bookmark", {"url": self.url, "title": "itest", "tags": ["X"]})
        self.assertFalse(saved["isError"])
        listed = self.session.call("list_bookmarks", {"tag": "x"})["structuredContent"]
        self.assertIn(self.url, [b["url"] for b in listed["items"]])
        self.assertTrue(state_keys("bookmarks-mcp", "bm/"))
        self.assertEqual(state_keys("feature-flags-mcp", "bm/"), [])

    def test_deleting_an_unknown_bookmark_is_a_tool_error(self) -> None:
        result = self.session.call("delete_bookmark", {"url": "https://itest.local/never"})
        self.assertTrue(result["isError"])


class FeatureFlagsLiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = StdioSession("feature-flags-mcp")
        self.session.request("initialize", {"protocolVersion": "2025-06-18"})
        self.flag = f"itest-{uuid.uuid4().hex[:12]}"

    def tearDown(self) -> None:
        self.session.call("delete_flag", {"name": self.flag})
        self.session.close()

    def test_rollout_splits_users_by_stable_buckets(self) -> None:
        self.session.call("set_flag", {"name": self.flag, "enabled": True, "rollout_percent": 30})
        results = [self.session.call("evaluate_flag", {"name": self.flag, "user_id": f"u{i}"})["structuredContent"] for i in range(40)]
        for r in results:
            self.assertEqual(r["on"], r["bucket"] < 30)
        self.assertTrue(any(r["on"] for r in results) and not all(r["on"] for r in results))

    def test_zero_percent_rollout_is_off_for_everyone_even_when_enabled(self) -> None:
        self.session.call("set_flag", {"name": self.flag, "enabled": True, "rollout_percent": 0})
        for i in range(10):
            self.assertFalse(self.session.call("evaluate_flag", {"name": self.flag, "user_id": f"u{i}"})["structuredContent"]["on"])

    def test_update_keeps_the_description(self) -> None:
        self.session.call("set_flag", {"name": self.flag, "enabled": True, "description": "keep me"})
        updated = self.session.call("set_flag", {"name": self.flag, "enabled": False})["structuredContent"]
        self.assertEqual(updated["description"], "keep me")

    def test_invalid_flag_name_is_rejected(self) -> None:
        self.assertTrue(self.session.call("set_flag", {"name": "Bad Name", "enabled": True})["isError"])


if __name__ == "__main__":
    unittest.main()
