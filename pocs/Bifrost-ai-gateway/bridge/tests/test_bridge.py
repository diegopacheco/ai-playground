import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge import build_prompt, route
from agent_sdk import ClaudeCodeAgent, OllamaAgent


class RecordingRunner:
    def __init__(self, output="pong\n", failure=None):
        self.commands = []
        self.output = output
        self.failure = failure

    def __call__(self, command):
        self.commands.append(command)
        if self.failure:
            raise self.failure
        return self.output


def chat(model, content="hello"):
    return {"model": model, "messages": [{"role": "user", "content": content}]}


class BridgeTest(unittest.TestCase):
    def test_bifrost_provider_path_selects_the_cli_that_runs(self):
        claude, ollama = RecordingRunner(), RecordingRunner()
        agents = {"claude": ClaudeCodeAgent(claude), "ollama": OllamaAgent(ollama)}
        status, body = route("POST", "/claude/v1/chat/completions", chat("claude-sonnet-5"), agents)
        self.assertEqual(200, status)
        self.assertEqual([["claude", "-p", "--model", "claude-sonnet-5", "hello"]], claude.commands)
        self.assertEqual([], ollama.commands)

    def test_cli_output_becomes_an_openai_chat_completion_bifrost_can_parse(self):
        agents = {"ollama": OllamaAgent(RecordingRunner("  pong\n"))}
        status, body = route("POST", "/ollama/v1/chat/completions", chat("llama3.2"), agents)
        self.assertEqual(200, status)
        self.assertEqual("chat.completion", body["object"])
        self.assertEqual({"role": "assistant", "content": "pong"}, body["choices"][0]["message"])
        self.assertEqual("stop", body["choices"][0]["finish_reason"])

    def test_provider_prefix_is_stripped_so_the_cli_gets_a_real_model_id(self):
        runner = RecordingRunner()
        route("POST", "/ollama/v1/chat/completions", chat("ollama-cli/tinyllama"), {"ollama": OllamaAgent(runner)})
        self.assertEqual(["ollama", "run", "tinyllama", "hello"], runner.commands[0])

    def test_a_single_user_message_is_sent_verbatim(self):
        self.assertEqual("What is 2+2?", build_prompt([{"role": "user", "content": "What is 2+2?"}]))

    def test_a_conversation_keeps_roles_so_the_cli_sees_the_context(self):
        prompt = build_prompt([
            {"role": "system", "content": "Be short"},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": ""},
        ])
        self.assertEqual("system: Be short\n\nuser: Hi", prompt)

    def test_cli_failure_is_reported_as_a_bad_gateway_not_a_fake_answer(self):
        agents = {"ollama": OllamaAgent(RecordingRunner(failure=RuntimeError("model not found")))}
        status, body = route("POST", "/ollama/v1/chat/completions", chat("missing"), agents)
        self.assertEqual(502, status)
        self.assertIn("model not found", body["error"]["message"])

    def test_empty_prompt_is_rejected_before_any_cli_runs(self):
        runner = RecordingRunner()
        status, _ = route("POST", "/ollama/v1/chat/completions", chat("llama3.2", "  "), {"ollama": OllamaAgent(runner)})
        self.assertEqual(400, status)
        self.assertEqual([], runner.commands)

    def test_streaming_is_refused_because_clis_return_the_whole_answer(self):
        runner = RecordingRunner()
        body = dict(chat("llama3.2"), stream=True)
        status, _ = route("POST", "/ollama/v1/chat/completions", body, {"ollama": OllamaAgent(runner)})
        self.assertEqual(400, status)
        self.assertEqual([], runner.commands)

    def test_unknown_provider_is_not_found(self):
        status, _ = route("POST", "/gpt/v1/chat/completions", chat("x"), {"ollama": OllamaAgent(RecordingRunner())})
        self.assertEqual(404, status)

    def test_models_endpoint_feeds_the_bifrost_model_catalog(self):
        status, body = route("GET", "/claude/v1/models", {}, {"claude": ClaudeCodeAgent(RecordingRunner())})
        self.assertEqual(200, status)
        self.assertIn("claude-sonnet-5", [model["id"] for model in body["data"]])


if __name__ == "__main__":
    unittest.main()
