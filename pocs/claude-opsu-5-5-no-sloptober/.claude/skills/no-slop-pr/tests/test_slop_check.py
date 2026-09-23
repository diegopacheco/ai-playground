import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import slop_check

FIX = HERE / "fixtures"


def read(name):
    return (FIX / name).read_text(encoding="utf-8")


def ids(result):
    return {hit["id"] for hit in result["hits"]}


class SlopCheckTest(unittest.TestCase):
    def test_agent_authored_pr_is_blocked(self):
        result = slop_check.score(read("slop.txt"), read("slop.diff"))
        self.assertEqual(result["verdict"], "BLOCK")
        self.assertIn("ai-coauthor", ids(result))
        self.assertIn("comment-cruft", ids(result))

    def test_terse_human_pr_passes(self):
        result = slop_check.score(read("human.txt"), read("human.diff"))
        self.assertEqual(result, {"score": 0, "verdict": "PASS", "hits": []})

    def test_slop_prose_without_trailer_is_only_suspect(self):
        result = slop_check.score(read("suspect.txt"))
        self.assertEqual(result["verdict"], "SUSPECT")
        self.assertNotIn("ai-coauthor", ids(result))

    def test_trailer_alone_blocks_because_it_admits_an_agent_wrote_it(self):
        result = slop_check.score("fix typo\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
        self.assertEqual(result["verdict"], "BLOCK")

    def test_human_coauthor_is_not_a_signal(self):
        result = slop_check.score("fix typo\n\nCo-Authored-By: Ana <ana@corp.dev>")
        self.assertEqual(result["verdict"], "PASS")

    def test_repeated_phrases_are_capped_so_one_word_cannot_block(self):
        result = slop_check.score("robust " * 50)
        self.assertEqual(result["score"], 30)
        self.assertEqual(result["verdict"], "SUSPECT")

    def test_comment_cruft_needs_enough_added_lines(self):
        self.assertEqual(slop_check.comment_ratio("+# a\n+# b\n"), 0.0)
        self.assertGreaterEqual(slop_check.comment_ratio(read("slop.diff")), 0.5)

    def test_cli_exit_code_fails_only_on_block(self):
        self.assertEqual(slop_check.main(["--text", str(FIX / "slop.txt"), "--diff", str(FIX / "slop.diff")]), 1)
        self.assertEqual(slop_check.main(["--text", str(FIX / "suspect.txt")]), 0)
        self.assertEqual(slop_check.main(["--text", str(FIX / "human.txt")]), 0)


if __name__ == "__main__":
    unittest.main()
