import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

SKILLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILLS / "decision-log" / "scripts"))

from decision import slug

GRADER = SKILLS / "answer-grader" / "scripts" / "grade.py"
DECISION = SKILLS / "decision-log" / "scripts" / "decision.py"


def run(script: Path, *args: str) -> tuple[int, Any]:
    completed = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=60)
    text = completed.stdout if completed.returncode == 0 else completed.stderr
    return completed.returncode, json.loads(text)


class SlugTest(unittest.TestCase):
    def test_titles_become_stable_ids_so_the_same_decision_is_not_recorded_twice(self) -> None:
        self.assertEqual(slug("Use Postgres RLS!"), "use-postgres-rls")
        self.assertEqual(slug("  use   postgres rls "), "use-postgres-rls")
        with self.assertRaises(ValueError):
            slug("!!!")


class DecisionLogLiveTest(unittest.TestCase):
    def test_lifecycle_add_supersede_and_status_filter(self) -> None:
        tag = uuid.uuid4().hex[:8]
        old, new = f"itest-{tag}-old", f"itest-{tag}-new"
        try:
            self.assertEqual(run(DECISION, "add", "--title", old, "--decision", "a", "--why", "b")[0], 0)
            self.assertEqual(run(DECISION, "add", "--title", new, "--decision", "c", "--why", "d")[0], 0)
            code, record = run(DECISION, "supersede", old, "--by", new)
            self.assertEqual((code, record["status"], record["superseded_by"]), (0, "superseded", new))
            _, accepted = run(DECISION, "list", "--status", "accepted")
            ids = [i["id"] for i in accepted["items"]]
            self.assertIn(new, ids)
            self.assertNotIn(old, ids)
        finally:
            run(DECISION, "remove", old)
            run(DECISION, "remove", new)

    def test_superseding_with_an_unknown_decision_fails_and_changes_nothing(self) -> None:
        title = f"itest-{uuid.uuid4().hex[:8]}"
        try:
            run(DECISION, "add", "--title", title, "--decision", "a", "--why", "b")
            code, _ = run(DECISION, "supersede", title, "--by", "does-not-exist")
            self.assertEqual(code, 1)
            self.assertEqual(run(DECISION, "show", title)[1]["status"], "accepted")
        finally:
            run(DECISION, "remove", title)


class AnswerGraderLiveTest(unittest.TestCase):
    def test_trend_and_best_score_follow_successive_gradings(self) -> None:
        suite = f"itest-{uuid.uuid4().hex[:8]}"
        checks = json.dumps([{"type": "contains", "value": "done"}])
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / "good.md", Path(tmp) / "bad.md"
            good.write_text("all done")
            bad.write_text("nothing")
            first = run(GRADER, "grade", "--suite", suite, "--answer", str(bad), "--checks", checks)[1]
            second = run(GRADER, "grade", "--suite", suite, "--answer", str(good), "--checks", checks)[1]
            third = run(GRADER, "grade", "--suite", suite, "--answer", str(bad), "--checks", checks)[1]
        try:
            self.assertEqual((first["trend"], first["score"]), ("first", 0.0))
            self.assertEqual(len(first["failed_checks"]), 1)
            self.assertEqual((second["trend"], second["score"]), ("better", 1.0))
            self.assertEqual((third["trend"], third["best"]), ("worse", 1.0))
            history = run(GRADER, "history", "--suite", suite)[1]
            self.assertEqual(history["scores"], [0.0, 1.0, 0.0])
        finally:
            subprocess.run(
                [str(SKILLS.parent / "infra" / "bin" / "agentinfra"), "--client", "answer-grader", "state", "delete", f"history/{suite}"],
                capture_output=True,
            )

    def test_empty_checks_are_refused(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".md") as answer:
            code, body = run(GRADER, "grade", "--suite", "s", "--answer", answer.name, "--checks", "[]")
        self.assertEqual(code, 1)
        self.assertIn("non empty", body["error"])


if __name__ == "__main__":
    unittest.main()
