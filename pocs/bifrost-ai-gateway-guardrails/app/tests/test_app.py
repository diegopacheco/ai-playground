import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import Gateway


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
        Gateway("http://127.0.0.1:8180/", post).ask("claude-cli", "claude-sonnet-5", "ping")
        url, payload = post.calls[0]
        self.assertEqual("http://127.0.0.1:8180/v1/chat/completions", url)
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

    def test_guardrail_block_is_shown_as_a_verdict_not_a_gateway_failure(self):
        post = FakePost(400, {"error": {"type": "guardrail_pii", "code": "email,us_ssn", "message": "blocked by guardrail pii: email, us_ssn"}, "extra_fields": {"provider": "claude-cli"}})
        result = Gateway("http://gw", post).ask("claude-cli", "claude-sonnet-5", "mail jane@acme.io")
        self.assertIsNone(result["answer"])
        self.assertEqual({"name": "pii", "action": "blocked", "matches": ["email", "us_ssn"], "message": "blocked by guardrail pii: email, us_ssn"}, result["guardrail"])

    def test_non_guardrail_400_is_still_a_gateway_error(self):
        post = FakePost(400, {"error": {"type": "invalid_request", "message": "bad model"}})
        with self.assertRaisesRegex(RuntimeError, "400: bad model"):
            Gateway("http://gw", post).ask("claude-cli", "claude-sonnet-5", "hi")

    def test_redacted_answer_reports_which_secrets_the_output_guardrail_masked(self):
        post = FakePost(body={"model": "llama3.2", "choices": [{"message": {"content": "a [REDACTED:aws_access_key] b [REDACTED:aws_access_key] [REDACTED:openai_key]"}}], "extra_fields": {"provider": "ollama-cli"}})
        result = Gateway("http://gw", post).ask("ollama-cli", "llama3.2", "hi")
        self.assertEqual("redacted", result["guardrail"]["action"])
        self.assertEqual(["aws_access_key", "openai_key"], result["guardrail"]["matches"])

    def test_clean_answer_has_no_guardrail(self):
        result = Gateway("http://gw", FakePost()).ask("claude-cli", "claude-sonnet-5", "ping")
        self.assertIsNone(result["guardrail"])

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


if __name__ == "__main__":
    unittest.main()
