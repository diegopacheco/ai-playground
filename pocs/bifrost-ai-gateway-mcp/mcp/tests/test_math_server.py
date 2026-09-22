import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math_server import CallLog, handle_body, handle_rpc


def rpc(method, params=None, request_id=1):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def call(log, name, **arguments):
    return handle_rpc(rpc("tools/call", {"name": name, "arguments": arguments}), log)["result"]


class MathServerTest(unittest.TestCase):
    def test_initialize_advertises_tools_so_bifrost_lists_them(self):
        result = handle_rpc(rpc("initialize", {"protocolVersion": "2025-06-18"}), CallLog())["result"]
        self.assertEqual("2025-06-18", result["protocolVersion"])
        self.assertIn("tools", result["capabilities"])
        self.assertEqual("math", result["serverInfo"]["name"])

    def test_tools_list_exposes_the_four_basic_operations_with_schemas(self):
        tools = handle_rpc(rpc("tools/list"), CallLog())["result"]["tools"]
        self.assertEqual(["add", "subtract", "multiply", "divide"], [tool["name"] for tool in tools])
        for tool in tools:
            self.assertEqual(["a", "b"], tool["inputSchema"]["required"])

    def test_each_operation_returns_the_exact_result(self):
        log = CallLog()
        self.assertEqual("7006652", call(log, "multiply", a=1234, b=5678)["content"][0]["text"])
        self.assertEqual("5", call(log, "add", a=2, b=3)["content"][0]["text"])
        self.assertEqual("-1", call(log, "subtract", a=2, b=3)["content"][0]["text"])
        self.assertEqual("2.5", call(log, "divide", a=5, b=2)["content"][0]["text"])

    def test_whole_float_results_read_as_integers_for_the_model(self):
        self.assertEqual({"result": 4}, call(CallLog(), "divide", a=8, b=2)["structuredContent"])

    def test_division_by_zero_is_a_tool_error_not_a_crash(self):
        result = call(CallLog(), "divide", a=10, b=0)
        self.assertTrue(result["isError"])
        self.assertEqual("division by zero", result["content"][0]["text"])

    def test_non_numeric_arguments_are_a_tool_error(self):
        self.assertTrue(call(CallLog(), "add", a="ten", b=1)["isError"])
        self.assertTrue(call(CallLog(), "add", a=True, b=1)["isError"])

    def test_numeric_strings_from_the_model_still_compute(self):
        self.assertEqual("12", call(CallLog(), "multiply", a="3", b="4")["content"][0]["text"])

    def test_unknown_tool_is_a_protocol_error(self):
        reply = handle_rpc(rpc("tools/call", {"name": "sqrt", "arguments": {"a": 4}}), CallLog())
        self.assertEqual(-32602, reply["error"]["code"])

    def test_every_call_is_logged_so_the_app_can_prove_mcp_ran(self):
        log = CallLog()
        call(log, "add", a=1, b=2)
        call(log, "divide", a=1, b=0)
        self.assertEqual(2, log.last())
        self.assertEqual([{"seq": 2, "tool": "divide", "arguments": {"a": 1, "b": 0}, "result": "division by zero", "error": True}], log.since(1))

    def test_notifications_get_no_reply(self):
        self.assertIsNone(handle_body({"jsonrpc": "2.0", "method": "notifications/initialized"}, CallLog()))

    def test_unknown_method_is_method_not_found(self):
        self.assertEqual(-32601, handle_rpc(rpc("resources/list"), CallLog())["error"]["code"])


if __name__ == "__main__":
    unittest.main()
