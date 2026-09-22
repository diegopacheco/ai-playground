import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import Gateway, SkillCatalog, skills_used


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


class SkillsUsedTest(unittest.TestCase):
    def test_the_signature_line_names_the_skill_that_answered(self):
        self.assertEqual(["acme-glossary"], skills_used("A blue freeze is a 36-hour deploy freeze.\n\nskill: acme-glossary"))

    def test_markdown_around_the_signature_still_counts(self):
        self.assertEqual(["acme-shipping-rates"], skills_used("Total: $23.80\n`skill: acme-shipping-rates`"))
        self.assertEqual(["acme-incident-report"], skills_used("**Skill: Acme-Incident-Report**"))

    def test_an_answer_without_a_signature_used_no_skill(self):
        self.assertEqual([], skills_used("I do not know what a blue freeze is. The skill: of cooking is useful."))

    def test_a_skill_signed_twice_is_listed_once(self):
        self.assertEqual(["acme-glossary", "acme-incident-report"], skills_used("skill: acme-glossary\nskill: acme-incident-report\nskill: acme-glossary"))

    def test_the_answer_reports_the_skills_bifrost_served_to_the_cli(self):
        post = FakePost(body={"model": "claude-sonnet-5", "choices": [{"message": {"content": "36 hours\nskill: acme-glossary"}}], "extra_fields": {"provider": "claude-cli"}})
        self.assertEqual(["acme-glossary"], Gateway("http://gw", post).ask("claude-cli", "claude-sonnet-5", "blue freeze?")["skills_used"])


class SkillCatalogTest(unittest.TestCase):
    def test_the_catalog_is_read_from_the_bifrost_skills_repository(self):
        urls = []

        def get(url):
            urls.append(url)
            return {"skills": [{"id": "1", "name": "acme-glossary", "description": "jargon", "latest_version": "1.0.0", "file_count": 0, "skill_md_body": "long"}]}

        self.assertEqual([{"name": "acme-glossary", "description": "jargon", "version": "1.0.0", "files": 0}], SkillCatalog("http://gw/", get).list())
        self.assertEqual(["http://gw/api/skills?limit=100&sort_by=name&order=asc"], urls)

if __name__ == "__main__":
    unittest.main()
