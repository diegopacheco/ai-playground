import io
import json
import os
import sys
import unittest
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills"))

from publish import local_skills

BIFROST = f"http://127.0.0.1:{os.environ.get('BIFROST_PORT', '8180')}"
APP = f"http://127.0.0.1:{os.environ.get('APP_PORT', '8192')}"

CLI_MODELS = [
    ("claude-cli", "claude-sonnet-5"),
    ("codex-cli", "gpt-5.6-luna"),
    ("agy-cli", "gemini-3.8-flash-low"),
    ("ollama-cli", "llama3.2"),
]

SKILL_QUESTIONS = [
    ("acme-glossary", "What is a blue freeze at Acme?", ["36"]),
    ("acme-shipping-rates", "How much does Acme charge to ship a 3.2 kg parcel to Zone B with express?", ["23.80"]),
    ("acme-incident-report", "Write an Acme incident report: login is failing for about 2500 customers since 10:05.", ["SEV-1", "team-gatekeeper"]),
]


def request(url, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    call = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(call, timeout=300) as response:
            raw = response.read()
            return response.status, raw if url.endswith(".zip") else json.loads(raw)
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, json.loads(failure.read() or b"{}")


class BifrostSkillsRepositoryTest(unittest.TestCase):
    def test_bifrost_serves_the_three_poc_skills_at_their_local_versions(self):
        status, body = request(f"{BIFROST}/api/skills?limit=100")
        self.assertEqual(200, status)
        served = {skill["name"]: skill["latest_version"] for skill in body["skills"]}
        self.assertEqual({skill["name"]: skill["version"] for skill in local_skills()}, served)

    def test_the_served_package_carries_the_skill_md_and_its_supporting_files(self):
        status, data = request(f"{BIFROST}/api/skills/serve/all/download.zip")
        self.assertEqual(200, status)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            self.assertIn("acme-shipping-rates/references/rates.md", names)
            self.assertIn('name: "acme-glossary"', archive.read("acme-glossary/SKILL.md").decode())

    def test_bifrost_lists_every_skill_as_a_claude_code_marketplace_plugin(self):
        status, body = request(f"{BIFROST}/api/skills/serve/claude-code/.claude-plugin/marketplace.json")
        self.assertEqual(200, status)
        plugins = {plugin["name"] for plugin in body["plugins"]}
        self.assertTrue({f"bifrost-{skill['name']}" for skill in local_skills()} | {"bifrost-all-skills"} <= plugins)


class AppUsesBifrostSkillsTest(unittest.TestCase):
    def test_each_cli_answers_with_facts_only_the_bifrost_skills_hold(self):
        cases = [(provider, model, skill, question, facts) for provider, model in CLI_MODELS[:3] for skill, question, facts in SKILL_QUESTIONS]
        with ThreadPoolExecutor(len(cases)) as pool:
            results = list(pool.map(lambda case: request(f"{APP}/api/ask", {"provider": case[0], "model": case[1], "question": case[3]}), cases))
        for (provider, _, skill, _, facts), (status, body) in zip(cases, results):
            with self.subTest(provider=provider, skill=skill):
                self.assertEqual(200, status, body)
                self.assertEqual(provider, body["provider"])
                for fact in facts:
                    self.assertIn(fact, body["answer"])
                self.assertEqual([skill], body["skills_used"])

    def test_ollama_gets_the_bifrost_skills_for_any_openai_client(self):
        status, body = request(f"{BIFROST}/v1/chat/completions", {"model": "ollama-cli/llama3.2", "messages": [{"role": "user", "content": "What is a blue freeze at Acme?"}]})
        self.assertEqual(200, status, body)
        self.assertEqual("ollama-cli", body["extra_fields"]["provider"])
        self.assertIn("36", body["choices"][0]["message"]["content"])

    def test_the_app_lists_the_skills_straight_from_bifrost(self):
        status, body = request(f"{APP}/api/skills")
        self.assertEqual(200, status)
        self.assertEqual(["acme-glossary", "acme-incident-report", "acme-shipping-rates"], [skill["name"] for skill in body["skills"]])
        self.assertEqual(1, next(skill for skill in body["skills"] if skill["name"] == "acme-shipping-rates")["files"])


class BifrostRoutingTest(unittest.TestCase):
    def test_bifrost_catalog_exposes_every_cli_provider_from_the_bridge(self):
        status, body = request(f"{BIFROST}/v1/models")
        self.assertEqual(200, status)
        ids = {model["id"] for model in body["data"]}
        for provider, _ in CLI_MODELS:
            self.assertTrue(any(model_id.startswith(f"{provider}/") for model_id in ids), provider)

    def test_cli_failure_surfaces_as_an_error_instead_of_an_answer(self):
        status, body = request(f"{APP}/api/ask", {"provider": "ollama-cli", "model": "no-such-model-xyz", "question": "hi"})
        self.assertEqual(502, status)
        self.assertIn("bifrost returned", body["error"])


if __name__ == "__main__":
    unittest.main()
