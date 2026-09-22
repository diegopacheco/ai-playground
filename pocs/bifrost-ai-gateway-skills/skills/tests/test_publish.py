import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from publish import SKILLS_DIR, load_skill, local_skills, publish


class FakeBifrost:
    def __init__(self, served=None, status=200):
        self.served = served or []
        self.status = status
        self.calls = []

    def __call__(self, method, url, payload=None):
        self.calls.append((method, url, payload))
        if method == "GET":
            return 200, {"skills": self.served}
        return self.status, {"skill": payload}


def skill(name="acme-glossary", version="1.0.0"):
    return {"name": name, "description": "d", "version": version, "skill_md_body": "b", "files": []}


class LoadSkillTest(unittest.TestCase):
    def test_the_poc_ships_exactly_the_three_skills_bifrost_serves(self):
        self.assertEqual(["acme-glossary", "acme-incident-report", "acme-shipping-rates"], [item["name"] for item in local_skills()])

    def test_supporting_files_travel_with_the_skill_so_the_cli_can_read_them(self):
        rates = load_skill(SKILLS_DIR / "acme-shipping-rates")
        self.assertEqual(["references/rates.md"], [item["path"] for item in rates["files"]])
        self.assertEqual("text/markdown", rates["files"][0]["mime_type"])
        self.assertIn("| B | 7.00 | 2.50 |", rates["files"][0]["content"])

    def test_source_files_are_not_uploaded_as_skill_files(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "my-skill"
            folder.mkdir()
            (folder / "skill.json").write_text(json.dumps({"description": "d", "version": "2.0.0"}))
            (folder / "body.md").write_text("Do the thing.")
            loaded = load_skill(folder)
        self.assertEqual({"name": "my-skill", "description": "d", "version": "2.0.0", "skill_md_body": "Do the thing.", "files": []}, loaded)

    def test_every_skill_signs_its_answers_so_the_app_can_prove_it_ran(self):
        for item in local_skills():
            self.assertIn(f"`skill: {item['name']}`", item["skill_md_body"])


class PublishTest(unittest.TestCase):
    def test_a_missing_skill_is_created_in_bifrost(self):
        bifrost = FakeBifrost()
        self.assertEqual([("acme-glossary", "created", "1.0.0")], publish("http://gw/", [skill()], bifrost))
        self.assertEqual(("POST", "http://gw/api/skills", skill()), bifrost.calls[1])

    def test_a_new_local_version_is_published_and_served(self):
        bifrost = FakeBifrost([{"id": "abc", "name": "acme-glossary", "latest_version": "1.0.0"}])
        self.assertEqual([("acme-glossary", "updated", "1.1.0")], publish("http://gw", [skill(version="1.1.0")], bifrost))
        method, url, payload = bifrost.calls[1]
        self.assertEqual(("PUT", "http://gw/api/skills/abc"), (method, url))
        self.assertTrue(payload["serve"])
        self.assertNotIn("name", payload)

    def test_restarting_the_stack_does_not_create_duplicate_versions(self):
        bifrost = FakeBifrost([{"id": "abc", "name": "acme-glossary", "latest_version": "1.0.0"}])
        self.assertEqual([("acme-glossary", "unchanged", "1.0.0")], publish("http://gw", [skill()], bifrost))
        self.assertEqual(1, len(bifrost.calls))

    def test_a_rejected_skill_fails_loud(self):
        with self.assertRaisesRegex(RuntimeError, "create skill acme-glossary returned 400"):
            publish("http://gw", [skill()], FakeBifrost(status=400))


if __name__ == "__main__":
    unittest.main()
