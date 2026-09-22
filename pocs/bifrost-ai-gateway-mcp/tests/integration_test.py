import json
import os
import unittest
import urllib.error
import urllib.request

BIFROST = f"http://127.0.0.1:{os.environ.get('BIFROST_PORT', '8080')}"
APP = f"http://127.0.0.1:{os.environ.get('APP_PORT', '8092')}"
MCP = f"http://127.0.0.1:{os.environ.get('MCP_PORT', '8093')}"

CLI_MODELS = [
    ("claude-cli", "claude-sonnet-5"),
    ("codex-cli", "gpt-5.6-luna"),
    ("agy-cli", "gemini-3.8-flash-low"),
    ("ollama-cli", "llama3.2"),
]


def request(url, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    call = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(call, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, json.loads(failure.read() or b"{}")


class BifrostRoutingTest(unittest.TestCase):
    def test_bifrost_catalog_exposes_every_cli_provider_from_the_bridge(self):
        status, body = request(f"{BIFROST}/v1/models")
        self.assertEqual(200, status)
        ids = {model["id"] for model in body["data"]}
        for provider, _ in CLI_MODELS:
            self.assertTrue(any(model_id.startswith(f"{provider}/") for model_id in ids), provider)

    def test_python_app_reaches_each_cli_only_through_bifrost(self):
        for provider, model in CLI_MODELS:
            with self.subTest(provider=provider):
                status, body = request(f"{APP}/api/ask", {"provider": provider, "model": model, "question": "Reply with exactly the word pong and nothing else"})
                self.assertEqual(200, status, body)
                self.assertEqual(provider, body["provider"])
                self.assertIn("pong", body["answer"].lower())

    def test_bifrost_openai_endpoint_serves_cli_models_to_any_openai_client(self):
        status, body = request(f"{BIFROST}/v1/chat/completions", {"model": "ollama-cli/tinyllama", "messages": [{"role": "user", "content": "Say hello"}]})
        self.assertEqual(200, status, body)
        self.assertEqual("ollama-cli", body["extra_fields"]["provider"])
        self.assertTrue(body["choices"][0]["message"]["content"].strip())

    def test_cli_failure_surfaces_as_an_error_instead_of_an_answer(self):
        status, body = request(f"{APP}/api/ask", {"provider": "ollama-cli", "model": "no-such-model-xyz", "question": "hi"})
        self.assertEqual(502, status)
        self.assertIn("bifrost returned", body["error"])


class BifrostMcpTest(unittest.TestCase):
    def test_bifrost_connects_the_math_mcp_server_and_discovers_its_tools(self):
        status, body = request(f"{BIFROST}/api/mcp/clients")
        self.assertEqual(200, status)
        math = next(client for client in body["clients"] if client["config"]["name"] == "math")
        self.assertEqual(["*"], math["config"]["tools_to_auto_execute"])
        self.assertEqual({"add", "subtract", "multiply", "divide"}, {tool["name"] for tool in math["tools"]})

    def test_app_math_question_makes_bifrost_run_the_mcp_tools_for_each_cli(self):
        for provider, model in CLI_MODELS[:3]:
            with self.subTest(provider=provider):
                status, body = request(f"{APP}/api/ask", {"provider": provider, "model": model, "question": "What is (1234 * 5678) + 91?"})
                self.assertEqual(200, status, body)
                self.assertIn("7006743", body["answer"].replace(",", ""))
                tools = [(call["tool"], call["result"]) for call in body["tool_calls"]]
                self.assertIn(("multiply", 7006652), tools)
                self.assertIn(("add", 7006743), tools)

    def test_bifrost_runs_the_mcp_loop_for_any_openai_client(self):
        _, before = request(f"{MCP}/calls?since=0")
        status, body = request(f"{BIFROST}/v1/chat/completions", {"model": "ollama-cli/llama3.2", "messages": [{"role": "user", "content": "What is 10 divided by 0?"}]})
        self.assertEqual(200, status, body)
        self.assertEqual("stop", body["choices"][0]["finish_reason"])
        _, after = request(f"{MCP}/calls?since={before['last']}")
        self.assertIn(("divide", "division by zero", True), [(call["tool"], call["result"], call["error"]) for call in after["calls"]])


if __name__ == "__main__":
    unittest.main()
