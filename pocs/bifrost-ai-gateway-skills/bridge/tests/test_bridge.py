import io
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge import SkillSync, build_prompt, route
from agent_sdk import AgyAgent, ClaudeCodeAgent, CodexAgent, OllamaAgent

GLOSSARY = {
    "acme-glossary/SKILL.md": '---\nname: "acme-glossary"\ndescription: "Acme jargon"\n---\nA blue freeze is a 36-hour deploy freeze.',
    "acme-glossary/references/terms.md": "owl shift: 22:00 to 06:00 UTC",
}


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


class FakeBifrost:
    def __init__(self, files=None, failure=None):
        self.files = files or {}
        self.failure = failure
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if self.failure:
            raise self.failure
        if not self.files:
            return b""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, content in self.files.items():
                archive.writestr(name, content)
        return buffer.getvalue()


def chat(model, content="hello"):
    return {"model": model, "messages": [{"role": "user", "content": content}]}


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.workspace = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def skills(self, bifrost=None):
        return SkillSync("http://gw/", self.workspace, bifrost or FakeBifrost())


class BridgeTest(WorkspaceTest):
    def test_bifrost_provider_path_selects_the_cli_that_runs(self):
        claude, ollama = RecordingRunner(), RecordingRunner()
        agents = {"claude": ClaudeCodeAgent(claude), "ollama": OllamaAgent(ollama)}
        status, _ = route("POST", "/claude/v1/chat/completions", chat("claude-sonnet-5"), agents, self.skills())
        self.assertEqual(200, status)
        self.assertEqual([["claude", "-p", "--model", "claude-sonnet-5", "hello"]], claude.commands)
        self.assertEqual([], ollama.commands)

    def test_cli_output_becomes_an_openai_chat_completion_bifrost_can_parse(self):
        agents = {"codex": CodexAgent(RecordingRunner("  pong\n"))}
        status, body = route("POST", "/codex/v1/chat/completions", chat("gpt-5.6-luna"), agents, self.skills())
        self.assertEqual(200, status)
        self.assertEqual("chat.completion", body["object"])
        self.assertEqual({"role": "assistant", "content": "pong"}, body["choices"][0]["message"])
        self.assertEqual("stop", body["choices"][0]["finish_reason"])

    def test_provider_prefix_is_stripped_so_the_cli_gets_a_real_model_id(self):
        runner = RecordingRunner()
        route("POST", "/ollama/v1/chat/completions", chat("ollama-cli/tinyllama"), {"ollama": OllamaAgent(runner)}, self.skills())
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
        status, body = route("POST", "/ollama/v1/chat/completions", chat("missing"), agents, self.skills())
        self.assertEqual(502, status)
        self.assertIn("model not found", body["error"]["message"])

    def test_empty_prompt_is_rejected_before_any_cli_runs(self):
        runner = RecordingRunner()
        status, _ = route("POST", "/ollama/v1/chat/completions", chat("llama3.2", "  "), {"ollama": OllamaAgent(runner)}, self.skills())
        self.assertEqual(400, status)
        self.assertEqual([], runner.commands)

    def test_streaming_is_refused_because_clis_return_the_whole_answer(self):
        runner = RecordingRunner()
        body = dict(chat("llama3.2"), stream=True)
        status, _ = route("POST", "/ollama/v1/chat/completions", body, {"ollama": OllamaAgent(runner)}, self.skills())
        self.assertEqual(400, status)
        self.assertEqual([], runner.commands)

    def test_unknown_provider_is_not_found(self):
        status, _ = route("POST", "/gpt/v1/chat/completions", chat("x"), {"ollama": OllamaAgent(RecordingRunner())}, self.skills())
        self.assertEqual(404, status)

    def test_models_endpoint_feeds_the_bifrost_model_catalog(self):
        status, body = route("GET", "/claude/v1/models", {}, {"claude": ClaudeCodeAgent(RecordingRunner())}, self.skills())
        self.assertEqual(200, status)
        self.assertIn("claude-sonnet-5", [model["id"] for model in body["data"]])


class SkillSyncTest(WorkspaceTest):
    def test_skills_served_by_bifrost_land_where_claude_code_and_codex_discover_them(self):
        bifrost = FakeBifrost(GLOSSARY)
        self.assertEqual(["acme-glossary"], self.skills(bifrost).sync())
        self.assertEqual(["http://gw/api/skills/serve/all/download.zip"], bifrost.urls)
        for root in (".claude/skills", ".agents/skills"):
            self.assertIn("36-hour", (self.workspace / root / "acme-glossary" / "SKILL.md").read_text())
            self.assertEqual("owl shift: 22:00 to 06:00 UTC", (self.workspace / root / "acme-glossary" / "references" / "terms.md").read_text())

    def test_a_skill_removed_in_bifrost_disappears_from_the_cli_workspace(self):
        bifrost = FakeBifrost(GLOSSARY)
        skills = self.skills(bifrost)
        skills.sync()
        bifrost.files = {"acme-other/SKILL.md": "other"}
        self.assertEqual(["acme-other"], skills.sync())
        self.assertFalse((self.workspace / ".claude/skills/acme-glossary").exists())

    def test_a_repository_without_skills_leaves_the_workspace_empty(self):
        self.assertEqual([], self.skills().sync())

    def test_an_archive_escaping_the_workspace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            self.skills(FakeBifrost({"../evil/SKILL.md": "x"})).sync()
        self.assertFalse((self.workspace.parent / "evil").exists())

    def test_every_request_syncs_so_a_newly_served_version_is_used_right_away(self):
        bifrost = FakeBifrost(GLOSSARY)
        agents = {"claude": ClaudeCodeAgent(RecordingRunner())}
        skills = self.skills(bifrost)
        route("POST", "/claude/v1/chat/completions", chat("claude-sonnet-5"), agents, skills)
        bifrost.files = {"acme-glossary/SKILL.md": "A blue freeze is now 48 hours."}
        route("POST", "/claude/v1/chat/completions", chat("claude-sonnet-5"), agents, skills)
        self.assertEqual("A blue freeze is now 48 hours.", (self.workspace / ".claude/skills/acme-glossary/SKILL.md").read_text())

    def test_bifrost_unreachable_is_a_bad_gateway_and_no_cli_runs(self):
        runner = RecordingRunner()
        status, body = route("POST", "/claude/v1/chat/completions", chat("claude-sonnet-5"), {"claude": ClaudeCodeAgent(runner)}, self.skills(FakeBifrost(failure=OSError("connection refused"))))
        self.assertEqual(502, status)
        self.assertIn("could not load skills from bifrost", body["error"]["message"])
        self.assertEqual([], runner.commands)

    def test_claude_and_codex_get_the_plain_question_and_find_skills_on_their_own(self):
        claude, codex = RecordingRunner(), RecordingRunner()
        agents = {"claude": ClaudeCodeAgent(claude), "codex": CodexAgent(codex)}
        skills = self.skills(FakeBifrost(GLOSSARY))
        route("POST", "/claude/v1/chat/completions", chat("claude-sonnet-5", "What is a blue freeze?"), agents, skills)
        route("POST", "/codex/v1/chat/completions", chat("gpt-5.6-luna", "What is a blue freeze?"), agents, skills)
        self.assertEqual("What is a blue freeze?", claude.commands[0][-1])
        self.assertEqual("What is a blue freeze?", codex.commands[0][-1])

    def test_agy_is_pointed_at_the_workspace_so_it_loads_the_skills(self):
        runner = RecordingRunner()
        route("POST", "/agy/v1/chat/completions", chat("gemini-3.8-flash-low"), {"agy": AgyAgent(runner)}, self.skills(FakeBifrost(GLOSSARY)))
        self.assertEqual(["agy", "--model", "gemini-3.8-flash-low", "--add-dir", str(self.workspace), "-p", "hello"], runner.commands[0])

    def test_ollama_has_no_skill_support_so_the_skills_travel_in_its_prompt(self):
        runner = RecordingRunner()
        route("POST", "/ollama/v1/chat/completions", chat("llama3.2", "What is a blue freeze?"), {"ollama": OllamaAgent(runner)}, self.skills(FakeBifrost(GLOSSARY)))
        prompt = runner.commands[0][-1]
        self.assertIn("=== skill acme-glossary ===", prompt)
        self.assertIn("--- file acme-glossary/references/terms.md ---\nowl shift", prompt)
        self.assertTrue(prompt.endswith("Question:\nWhat is a blue freeze?"))

    def test_health_lists_the_skills_the_clis_can_use(self):
        skills = self.skills(FakeBifrost(GLOSSARY))
        skills.sync()
        _, body = route("GET", "/health", {}, {}, skills)
        self.assertEqual(["acme-glossary"], body["skills"])


if __name__ == "__main__":
    unittest.main()
