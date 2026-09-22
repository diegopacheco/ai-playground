import json
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
        self.assertEqual(["ollama", "run", "--nowordwrap", "tinyllama", "hello"], runner.commands[0])

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


TOOLS = [{"type": "function", "function": {"name": "multiply", "description": "a * b", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}}]


def tool_chat(messages):
    return {"model": "claude-sonnet-5", "messages": messages, "tools": TOOLS}


class BridgeToolTest(unittest.TestCase):
    def test_mcp_tools_bifrost_injects_are_described_to_the_cli(self):
        runner = RecordingRunner()
        route("POST", "/claude/v1/chat/completions", tool_chat([{"role": "user", "content": "3 times 4?"}]), {"claude": ClaudeCodeAgent(runner)})
        prompt = runner.commands[0][-1]
        self.assertIn('"name": "multiply"', prompt)
        self.assertIn("never do the math yourself", prompt)
        self.assertTrue(prompt.endswith("Conversation:\n\n3 times 4?"))

    def test_cli_json_reply_becomes_an_openai_tool_call_so_bifrost_runs_the_mcp_tool(self):
        runner = RecordingRunner('```json\n{"tool_calls": [{"name": "multiply", "arguments": {"a": 3, "b": 4}}]}\n```')
        status, body = route("POST", "/claude/v1/chat/completions", tool_chat([{"role": "user", "content": "3 times 4?"}]), {"claude": ClaudeCodeAgent(runner)})
        self.assertEqual(200, status)
        choice = body["choices"][0]
        self.assertEqual("tool_calls", choice["finish_reason"])
        self.assertIsNone(choice["message"]["content"])
        call = choice["message"]["tool_calls"][0]
        self.assertEqual("function", call["type"])
        self.assertEqual("multiply", call["function"]["name"])
        self.assertEqual({"a": 3, "b": 4}, json.loads(call["function"]["arguments"]))

    def test_a_tool_the_gateway_never_offered_is_not_called(self):
        runner = RecordingRunner('{"tool_calls": [{"name": "rm_rf", "arguments": {}}]}')
        _, body = route("POST", "/claude/v1/chat/completions", tool_chat([{"role": "user", "content": "hi"}]), {"claude": ClaudeCodeAgent(runner)})
        self.assertEqual("stop", body["choices"][0]["finish_reason"])
        self.assertNotIn("tool_calls", body["choices"][0]["message"])

    def test_plain_text_after_tool_results_is_the_final_answer(self):
        runner = RecordingRunner("12\n")
        _, body = route("POST", "/claude/v1/chat/completions", tool_chat([{"role": "user", "content": "3 times 4?"}]), {"claude": ClaudeCodeAgent(runner)})
        self.assertEqual({"role": "assistant", "content": "12"}, body["choices"][0]["message"])

    def test_tool_results_from_the_agent_loop_reach_the_cli_as_context(self):
        prompt = build_prompt([
            {"role": "user", "content": "3 times 4?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "multiply", "arguments": '{"a": 3, "b": 4}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "12"},
        ])
        self.assertIn('assistant: called tool multiply with {"a": 3, "b": 4}', prompt)
        self.assertIn("tool result: 12", prompt)

    def test_ollama_runs_without_word_wrap_so_tool_call_json_is_not_broken_by_terminal_escapes(self):
        runner = RecordingRunner('{"tool_calls": [{"name": "multiply", "arguments": {"a": 3, "b": 4}}]}')
        _, body = route("POST", "/ollama/v1/chat/completions", dict(tool_chat([{"role": "user", "content": "3 times 4?"}]), model="llama3.2"), {"ollama": OllamaAgent(runner)})
        self.assertEqual(["ollama", "run", "--nowordwrap", "llama3.2"], runner.commands[0][:4])
        self.assertEqual("tool_calls", body["choices"][0]["finish_reason"])

    def test_requests_without_tools_keep_the_plain_prompt(self):
        runner = RecordingRunner()
        route("POST", "/claude/v1/chat/completions", chat("claude-sonnet-5", "hello"), {"claude": ClaudeCodeAgent(runner)})
        self.assertEqual("hello", runner.commands[0][-1])


if __name__ == "__main__":
    unittest.main()
