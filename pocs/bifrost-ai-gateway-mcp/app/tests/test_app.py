import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import Gateway, ToolLog


class FakePost:
    def __init__(self, status=200, body=None):
        self.calls = []
        self.status = status
        self.body = body if body is not None else {
            "model": "claude-sonnet-5",
            "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            "extra_fields": {"provider": "claude-cli"},
        }

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.status, self.body


class GatewayTest(unittest.TestCase):
    def test_every_question_goes_to_bifrost_with_a_provider_prefixed_model(self):
        post = FakePost()
        Gateway("http://127.0.0.1:8080/", post).ask("claude-cli", "claude-sonnet-5", "ping")
        url, payload = post.calls[0]
        self.assertEqual("http://127.0.0.1:8080/v1/chat/completions", url)
        self.assertEqual("claude-cli/claude-sonnet-5", payload["model"])
        self.assertEqual([{"role": "user", "content": "ping"}], payload["messages"])

    def test_answer_reports_the_provider_bifrost_actually_routed_to(self):
        post = FakePost(body={"model": "llama3.2", "choices": [{"message": {"content": "hi"}}], "extra_fields": {"provider": "ollama-cli"}})
        result = Gateway("http://gw", post).ask("ollama-cli", "llama3.2", "hi")
        self.assertEqual("hi", result["answer"])
        self.assertEqual("ollama-cli", result["provider"])
        self.assertIsInstance(result["latency_ms"], int)

    def test_gateway_error_is_raised_with_bifrost_message(self):
        post = FakePost(502, {"error": {"message": "Command failed"}})
        with self.assertRaisesRegex(RuntimeError, "502: Command failed"):
            Gateway("http://gw", post).ask("codex-cli", "gpt-5.6-luna", "hi")

    def test_unknown_provider_never_reaches_bifrost(self):
        post = FakePost()
        with self.assertRaisesRegex(ValueError, "unknown provider"):
            Gateway("http://gw", post).ask("openai", "gpt-4o", "hi")
        self.assertEqual([], post.calls)

    def test_blank_question_never_reaches_bifrost(self):
        post = FakePost()
        with self.assertRaisesRegex(ValueError, "question is required"):
            Gateway("http://gw", post).ask("ollama-cli", "llama3.2", "  ")
        self.assertEqual([], post.calls)


class FakeGet:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, url):
        since = int(url.split("since=")[1])
        return {"last": len(self.calls), "calls": [call for call in self.calls if call["seq"] > since]}


class ToolLogTest(unittest.TestCase):
    def test_answer_lists_only_the_mcp_calls_made_during_this_question(self):
        old = {"seq": 1, "tool": "add", "arguments": {"a": 1, "b": 1}, "result": 2, "error": False}
        new = {"seq": 2, "tool": "multiply", "arguments": {"a": 3, "b": 4}, "result": 12, "error": False}
        get = FakeGet([old])

        def post(url, payload):
            get.calls.append(new)
            return 200, {"model": "claude-sonnet-5", "choices": [{"message": {"content": "12"}}], "extra_fields": {"provider": "claude-cli"}}

        result = Gateway("http://gw", post, ToolLog("http://mcp", get)).ask("claude-cli", "claude-sonnet-5", "3 times 4?")
        self.assertEqual([{"tool": "multiply", "arguments": {"a": 3, "b": 4}, "result": 12, "error": False}], result["tool_calls"])

    def test_without_a_tool_log_the_answer_has_no_tool_calls(self):
        self.assertEqual([], Gateway("http://gw", FakePost()).ask("claude-cli", "claude-sonnet-5", "hi")["tool_calls"])


if __name__ == "__main__":
    unittest.main()
